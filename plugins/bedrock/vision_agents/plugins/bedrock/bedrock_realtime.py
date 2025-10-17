import asyncio
import json
import logging
from typing import Optional, List, Dict, Any
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.track_util import PcmData
import boto3
from botocore.exceptions import ClientError

from vision_agents.core.edge.types import Participant
from vision_agents.core.llm import realtime
from vision_agents.core.llm.events import RealtimeAudioOutputEvent, LLMResponseChunkEvent
from vision_agents.core.llm.llm_types import ToolSchema, NormalizedToolCallItem
from vision_agents.core.processors import Processor
from vision_agents.core.utils.utils import frame_to_png_bytes
import av

from vision_agents.core.utils.video_forwarder import VideoForwarder
from . import events

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "us.amazon.nova-sonic-v1:0"
DEFAULT_SAMPLE_RATE = 16000


class Realtime(realtime.Realtime):
    """
    Realtime on AWS Bedrock with support for audio/video streaming.
    
    This implementation uses AWS Bedrock's Converse Stream API for real-time
    interactions with Amazon Nova Sonic, which is specifically designed for
    speech-to-speech conversations with ultra-low latency.
    
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
    
    Development notes:
    - Audio data should be PCM format, 16-bit, mono
    - Output audio is typically at 24kHz or 16kHz depending on the model
    - Input audio is resampled if needed
    """
    
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
        
        # Initialize boto3 bedrock-runtime client
        session_kwargs = {"region_name": region_name}
        if aws_access_key_id:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token
            
        self.client = boto3.client("bedrock-runtime", **session_kwargs)
        self.logger = logging.getLogger(__name__)
        
        # Audio output track - Bedrock typically outputs at 16kHz
        self.output_track = AudioStreamTrack(
            framerate=sample_rate, stereo=False, format="s16"
        )
        
        self._video_forwarder: Optional[VideoForwarder] = None
        self._stream_task: Optional[asyncio.Task[Any]] = None
        self._connected = False
        self._message_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._conversation_messages: List[Dict[str, Any]] = []
        self._pending_tool_uses: Dict[int, Dict[str, Any]] = {}  # Track tool calls across stream events

    async def simple_response(
        self, 
        text: str, 
        processors: Optional[List[Processor]] = None,
        participant: Optional[Participant] = None
    ):
        """
        Simple response standardizes how to send a text instruction to this LLM.
        
        Args:
            text: The text instruction to send
            processors: Optional list of processors
            participant: Optional participant object
        
        Example:
            await llm.simple_response("tell me a poem about Boulder")
        """
        self.logger.info("Simple response called with text: %s", text)
        await self._send_message({
            "role": "user",
            "content": [{"text": text}]
        })

    async def simple_audio_response(self, pcm: PcmData):
        """
        Simple audio response standardizes how to send audio to the LLM.
        
        Args:
            pcm: PCM audio data
        
        Example:
            pcm = PcmData(...)
            await llm.simple_audio_response(pcm)
        """
        if not self._connected:
            return

        self.logger.debug(f"Sending audio to Bedrock: {pcm.duration}s")
        
        # Convert PCM to bytes
        audio_bytes = pcm.samples.tobytes()
        
        await self._send_message({
            "role": "user",
            "content": [{
                "audio": {
                    "format": "pcm",
                    "source": {
                        "bytes": audio_bytes
                    }
                }
            }]
        })

    async def connect(self):
        """
        Connect to Bedrock's streaming service.
        """
        self.logger.info("Connecting to Bedrock realtime with model %s", self.model)
        self._connected = True
        
        # Start the streaming task
        self._stream_task = asyncio.create_task(self._stream_loop())
        
        self.logger.info("Bedrock realtime connected")

    async def _stream_loop(self):
        """
        Main loop for streaming with Bedrock.
        This continuously processes messages from the queue and sends them via Converse Stream.
        """
        try:
            while self._connected:
                # Wait for messages or timeout
                try:
                    message = await asyncio.wait_for(
                        self._message_queue.get(), 
                        timeout=0.1
                    )
                    
                    # Add to conversation history
                    self._conversation_messages.append(message)
                    
                    # Call Bedrock Converse Stream
                    await self._process_stream_response()
                    
                except asyncio.TimeoutError:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Stream loop error: {e}")
            raise
        finally:
            self.logger.info("Stream loop ended")

    async def _process_stream_response(self):
        """
        Process a streaming response from Bedrock.
        """
        try:
            # Build the system instruction
            system_instructions = [{"text": self._build_enhanced_instructions()}]
            
            # Add tools if available
            kwargs: Dict[str, Any] = {
                "modelId": self.model,
                "messages": self._conversation_messages,
                "system": system_instructions,
            }
            
            # Add tools configuration
            tools = self.get_available_functions()
            if tools:
                kwargs["toolConfig"] = {
                    "tools": self._convert_tools_to_provider_format(tools)
                }
            
            # Call Converse Stream
            response = self.client.converse_stream(**kwargs)
            stream = response.get('stream')
            
            # Collect the assistant's response
            assistant_content: List[Dict[str, Any]] = []
            text_parts: List[str] = []
            tool_calls: List[NormalizedToolCallItem] = []
            
            for event in stream:
                # Emit the raw event
                self.events.send(events.BedrockStreamEvent(
                    plugin_name="bedrock",
                    event_data=event
                ))
                
                # Handle contentBlockStart - initialize tool use tracking
                if 'contentBlockStart' in event:
                    start = event['contentBlockStart'].get('start', {})
                    idx = event['contentBlockStart'].get('contentBlockIndex', 0)
                    
                    if 'toolUse' in start:
                        tool_use = start['toolUse']
                        tool_use_id = tool_use.get('toolUseId', '')
                        tool_name = tool_use.get('name', '')
                        
                        self.logger.info(f"Tool call started: {tool_name} (id: {tool_use_id})")
                        
                        # Initialize tracking for this tool call
                        self._pending_tool_uses[idx] = {
                            "id": tool_use_id,
                            "name": tool_name,
                            "input_parts": []
                        }
                
                # Handle contentBlockDelta - accumulate data
                if 'contentBlockDelta' in event:
                    delta = event['contentBlockDelta']['delta']
                    idx = event['contentBlockDelta'].get('contentBlockIndex', 0)
                    
                    # Handle text content
                    if 'text' in delta:
                        text = delta['text']
                        text_parts.append(text)
                        self.logger.info(f"Bedrock output: {text}")
                        
                        # Emit chunk event
                        self.events.send(LLMResponseChunkEvent(
                            plugin_name="bedrock",
                            content_index=idx,
                            item_id="",
                            output_index=0,
                            sequence_number=0,
                            delta=text,
                        ))
                    
                    # Handle audio content
                    elif 'audio' in delta:
                        audio_data = delta['audio'].get('bytes')
                        if audio_data:
                            # Emit audio output event
                            audio_event = RealtimeAudioOutputEvent(
                                plugin_name="bedrock",
                                audio_data=audio_data,
                                sample_rate=self.sample_rate
                            )
                            self.events.send(audio_event)
                            
                            # Write to output track
                            await self.output_track.write(audio_data)
                    
                    # Handle tool use input (streaming JSON)
                    elif 'toolUse' in delta:
                        if idx in self._pending_tool_uses:
                            input_chunk = delta['toolUse'].get('input', '')
                            if input_chunk:
                                self._pending_tool_uses[idx]['input_parts'].append(input_chunk)
                
                # Handle contentBlockStop - finalize tool calls
                if 'contentBlockStop' in event:
                    idx = event['contentBlockStop'].get('contentBlockIndex', 0)
                    
                    # Check if this was a tool use block
                    if idx in self._pending_tool_uses:
                        pending = self._pending_tool_uses.pop(idx)
                        
                        # Parse the accumulated input JSON
                        input_json = "".join(pending['input_parts']).strip() or "{}"
                        try:
                            args = json.loads(input_json)
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to parse tool input JSON: {e}")
                            args = {}
                        
                        # Create normalized tool call
                        tool_call: NormalizedToolCallItem = {
                            "type": "tool_call",
                            "id": pending["id"],
                            "name": pending["name"],
                            "arguments_json": args
                        }
                        tool_calls.append(tool_call)
                        self.logger.info(f"Tool call completed: {pending['name']} with args: {args}")
            
            # Add assistant response to conversation
            if text_parts:
                assistant_content.append({"text": "".join(text_parts)})
            
            if assistant_content:
                self._conversation_messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
            
            # Handle tool calls if any
            if tool_calls:
                await self._handle_tool_calls(tool_calls)
                
        except ClientError as e:
            self.logger.error(f"Bedrock stream error: {e}")

    async def _handle_tool_calls(self, tool_calls: List[NormalizedToolCallItem]):
        """
        Handle tool calls from Bedrock by executing them and sending results back.
        
        This follows the Bedrock Converse API format for tool results.
        See: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html
        
        Args:
            tool_calls: List of tool calls to execute
        """
        try:
            self.logger.info(f"Handling {len(tool_calls)} tool call(s)")
            
            # Execute all tool calls concurrently
            tool_results = []
            for tc in tool_calls:
                self.logger.info(f"Executing tool: {tc['name']} with args: {tc['arguments_json']}")
                
                # Execute the tool
                _, result, error = await self._run_one_tool(tc, timeout_s=30)
                
                # Format the result
                if error:
                    self.logger.error(f"Tool {tc['name']} failed: {error}")
                    result_content = {"error": str(error)}
                    status = "error"
                else:
                    self.logger.info(f"Tool {tc['name']} succeeded: {result}")
                    # Ensure result is JSON-serializable
                    if isinstance(result, dict):
                        result_content = result
                    else:
                        result_content = {"result": str(result)}
                    status = "success"
                
                # Build tool result in Bedrock format
                tool_result = {
                    "toolUseId": tc["id"],
                    "content": [{"json": result_content}],
                    "status": status
                }
                tool_results.append(tool_result)
            
            # Send tool results back as a user message
            tool_result_message = {
                "role": "user",
                "content": [{"toolResult": tr} for tr in tool_results]
            }
            
            self.logger.info(f"Sending {len(tool_results)} tool result(s) back to Bedrock")
            await self._send_message(tool_result_message)
            
        except Exception as e:
            self.logger.error(f"Error handling tool calls: {e}", exc_info=True)

    async def _send_message(self, message: Dict[str, Any]):
        """
        Queue a message to be sent to Bedrock.
        
        Args:
            message: Message in Bedrock format
        """
        await self._message_queue.put(message)

    async def _close_impl(self):
        """
        Close the Bedrock connection.
        """
        self._connected = False
        
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None

    async def _watch_video_track(self, track: Any, **kwargs) -> None:
        """
        Start sending video frames to Bedrock using VideoForwarder.
        
        Args:
            track: Video track to watch
            **kwargs: Additional arguments including optional shared_forwarder
        """
        shared_forwarder = kwargs.get('shared_forwarder')
        
        if self._video_forwarder is not None and shared_forwarder is None:
            self.logger.warning("Video sender already running, stopping previous one")
            await self._stop_watching_video_track()
        
        if shared_forwarder is not None:
            # Use the shared forwarder
            self._video_forwarder = shared_forwarder
            self.logger.info(f"🎥 Bedrock subscribing to shared VideoForwarder at {self.fps} FPS")
            await self._video_forwarder.start_event_consumer(
                self._send_video_frame,
                fps=float(self.fps),
                consumer_name="bedrock"
            )
        else:
            # Create our own VideoForwarder
            self._video_forwarder = VideoForwarder(
                track,  # type: ignore[arg-type]
                max_buffer=5,
                fps=float(self.fps),
                name="bedrock_forwarder",
            )
            
            await self._video_forwarder.start()
            await self._video_forwarder.start_event_consumer(self._send_video_frame)
            
            self.logger.info(f"Started video forwarding with {self.fps} FPS")

    async def _stop_watching_video_track(self) -> None:
        """Stop watching the video track."""
        if self._video_forwarder is not None:
            await self._video_forwarder.stop()
            self._video_forwarder = None
            self.logger.info("Stopped video forwarding")

    async def _send_video_frame(self, frame: av.VideoFrame) -> None:
        """
        Send a video frame to Bedrock.
        
        Args:
            frame: Video frame to send
        """
        if not frame:
            return

        try:
            # Convert frame to image bytes (PNG or JPEG)
            image_bytes = frame_to_png_bytes(frame)
            
            # Send as a message with image content
            await self._send_message({
                "role": "user",
                "content": [{
                    "image": {
                        "format": "png",
                        "source": {
                            "bytes": image_bytes
                        }
                    }
                }]
            })
        except Exception as e:
            self.logger.error(f"Error sending video frame: {e}")

    def _convert_tools_to_provider_format(self, tools: List[ToolSchema]) -> List[Dict[str, Any]]:
        """
        Convert ToolSchema objects to Bedrock format.
        
        Args:
            tools: List of ToolSchema objects
            
        Returns:
            List of tools in Bedrock format
        """
        bedrock_tools = []
        for tool in tools:
            bedrock_tool = {
                "toolSpec": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "inputSchema": {
                        "json": tool["parameters_schema"]
                    }
                }
            }
            bedrock_tools.append(bedrock_tool)
        return bedrock_tools

    def _extract_tool_calls_from_event(self, event: Dict[str, Any]) -> List[NormalizedToolCallItem]:
        """
        Extract tool calls from Bedrock streaming event.
        
        Note: This method is now largely superseded by the inline tool call
        accumulation in _process_stream_response. It's kept for compatibility
        but the main logic is handled during stream processing.
        
        Args:
            event: Bedrock streaming event
            
        Returns:
            List of normalized tool call items (typically empty as they're 
            handled inline during streaming)
        """
        # Tool calls are now handled inline during stream processing
        # by tracking contentBlockStart, contentBlockDelta, and contentBlockStop events
        return []

