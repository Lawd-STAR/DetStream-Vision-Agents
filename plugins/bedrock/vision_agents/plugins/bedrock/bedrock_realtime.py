import asyncio
import base64
import json
import logging
from typing import Optional, List, Dict, Any
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.track_util import PcmData

from vision_agents.core.edge.types import Participant
from vision_agents.core.llm import realtime
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

from vision_agents.core.utils.video_forwarder import VideoForwarder
from . import events

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "us.amazon.nova-sonic-v1:0"
DEFAULT_SAMPLE_RATE = 16000


class Realtime(realtime.Realtime):
    """
    Realtime on AWS Bedrock with support for audio/video streaming.

    A few things are different about Nova compared to other STS solutions

        1. two init events. there is a session start and a prompt start
        2. promptName basically works like a unique identifier. it's created client side and sent to nova
        3. input/text events are wrapped. so its common to do start event, text event, stop event
        4. on close there is an session and a prompt end event

    AWS Nova samples are the best docs:

        simple: https://github.com/aws-samples/amazon-nova-samples/blob/main/speech-to-speech/sample-codes/console-python/nova_sonic_simple.py
        full: https://github.com/aws-samples/amazon-nova-samples/blob/main/speech-to-speech/sample-codes/console-python/nova_sonic.py
        tool use: https://github.com/aws-samples/amazon-nova-samples/blob/main/speech-to-speech/sample-codes/console-python/nova_sonic_tool_use.py

    Available voices are documented here:
    https://docs.aws.amazon.com/nova/latest/userguide/available-voices.html
    
    Examples:
    
        from vision_agents.plugins import bedrock
        
        llm = bedrock.Realtime(
            model="us.amazon.nova-sonic-v1:0",
            region_name="us-east-1"
        )
        
        # Connect to the session
        await llm.connect()
        
        # Simple text response
        await llm.simple_response("Describe what you see and say hi")
        
        # Send audio
        await llm.simple_audio_response(pcm_data)
        
        # Close when done
        await llm.close()
    """
    connected : bool = False

    async def _close_impl(self):
        pass

    def __init__(
        self, 
        model: str = DEFAULT_MODEL,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        sample_rate: int = 16000,
        **kwargs
    ) -> None:
        """
        Initialize Bedrock Realtime with Nova Sonic.
        
        Args:
            model: The Bedrock model ID (default: us.amazon.nova-sonic-v1:0)
            region_name: AWS region name (default: us-east-1)
            aws_access_key_id: Optional AWS access key ID
            aws_secret_access_key: Optional AWS secret access key
            aws_session_token: Optional AWS session token
            sample_rate: Audio sample rate in Hz (default: 16000)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.model = model
        self.region_name = region_name
        self.sample_rate = sample_rate
        
        # Initialize Bedrock Runtime client with SDK
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{region_name}.amazonaws.com",
            region=region_name,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self.client = BedrockRuntimeClient(config=config)
        self.logger = logging.getLogger(__name__)
        
        # Audio output track - Bedrock typically outputs at 16kHz
        self.output_track = AudioStreamTrack(
            framerate=sample_rate, stereo=False, format="s16"
        )
        
        self._video_forwarder: Optional[VideoForwarder] = None
        self._stream_task: Optional[asyncio.Task[Any]] = None
        self._is_connected = False
        self._message_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._conversation_messages: List[Dict[str, Any]] = []
        self._pending_tool_uses: Dict[int, Dict[str, Any]] = {}  # Track tool calls across stream events
        
        # Audio streaming configuration
        self.prompt_name = "default_prompt"
        self.audio_content_name = "audio_input"

    async def connect(self):
        """To connect we need to do a few things

        - start a bi directional stream
        - send session start event
        - send prompt start event
        - send text content start, text content, text content end

        Two unusual things here are that you have
        - 2 init events (session and prompt start)
        - text content is wrapped

        The init events should be easy to customize
        """
        if self._is_connected:
            self.logger.warning("Already connected")
            return
            
        # Create the input stream operation
        input_operation = InvokeModelWithBidirectionalStreamOperationInput(
            model_id=self.model
        )
        
        # Start the bidirectional stream
        self.stream = await self.client.invoke_model_with_bidirectional_stream(input_operation)
        
        self.connected = True
        self.logger.info(f"Connected to Bedrock model: {self.model}")
        
        # Start processing the output stream asynchronously
        self._stream_task = asyncio.create_task(self._process_output_stream())

        # audio input is always on
        await self.start_audio_input()

    async def send_event(self, event_json):
        """Send an event to the stream."""
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        await self.stream.input_stream.send(event)

    async def start_audio_input(self):
        """Start audio input stream."""
        audio_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.audio_content_name}",
                    "type": "AUDIO",
                    "interactive": true,
                    "role": "USER",
                    "audioInputConfiguration": {{
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 48000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64"
                    }}
                }}
            }}
        }}
        '''
        await self.send_event(audio_content_start)

    @property
    def is_active(self) -> bool:
        """Return True if the session is active and ready to send/receive."""
        return self._is_connected

    async def send_audio_chunk(self, audio_bytes):
        """Send an audio chunk to the stream."""
        if not self.is_active:
            return
            
        blob = base64.b64encode(audio_bytes)
        audio_event = f'''
        {{
            "event": {{
                "audioInput": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.audio_content_name}",
                    "content": "{blob.decode('utf-8')}"
                }}
            }}
        }}
        '''
        await self.send_event(audio_event)

    async def simple_audio_response(self, pcm: PcmData):
        """Send audio data to the model for processing."""
        if not self.is_active:
            self.logger.warning("Cannot send audio: session not active")
            return
            
        # Convert PCM samples to bytes
        audio_bytes = pcm.samples.tobytes()
        
        # Send the audio chunk
        await self.send_audio_chunk(audio_bytes)
        
        # Emit audio input event for tracking
        self._emit_audio_input_event(audio_bytes, sample_rate=pcm.sample_rate)

    async def _process_output_stream(self):
        """Process the output stream from the model."""
        try:
            async for chunk in self.stream.output_stream:
                if chunk.value and chunk.value.bytes_:
                    # Decode the JSON payload
                    payload_json = chunk.value.bytes_.decode('utf-8')
                    try:
                        payload = json.loads(payload_json)
                        await self._handle_output_payload(payload)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to decode JSON payload: {e}")
        except Exception as e:
            self.logger.error(f"Error processing output stream: {e}")
            self._is_connected = False

    async def _handle_output_payload(self, payload: Dict[str, Any]):
        """Handle the output payload from the model."""
        # Add payload to message queue for processing
        await self._message_queue.put(payload)
