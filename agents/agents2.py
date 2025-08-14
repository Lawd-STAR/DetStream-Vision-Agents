import asyncio
import logging
import traceback
from typing import Optional, Callable, List
from uuid import uuid4

from getstream.video import rtc
from getstream.video.rtc import audio_track
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig, TrackType

from agents.agents import (
    TransformedVideoTrack,
    Tool,
    PreProcessor, 
    Model,
    TurnDetection,
    ImageProcessor,
    VideoTransformer,
    STT,
    TTS,
    VAD,
    STS
)

"""
TODO:
- improve naming on interval
- how to ping the LLM when connection is ready?
"""


class Agent:
    """
    AI Agent that can join Stream video calls and interact with participants.

    Note that the agent can run in several different modes:
    - STS Model (Speech-to-Speech with OpenAI Realtime API)
    - STT -> Model -> TTS (Traditional pipeline)
    - Video AI/coach
    - Video transformation

    With either a mix or match of those.

    Example usage:
        # Traditional STT -> Model -> TTS pipeline
        agent = Agent(
            instructions="Roast my in-game performance in a funny but encouraging manner",
            pre_processors=[Roboflow(), dota_api("gameid")],
            model=openai_model,
            stt=speech_to_text,
            tts=text_to_speech,
            turn_detection=turn_detector
        )

        # OpenAI Realtime STS mode
        agent = Agent(
            instructions="You are a helpful assistant",
            sts_model=openai_realtime_sts
        )

        await agent.join(call)
    """

    def start_interval(self):
        """Start the interval processing task for image/video analysis."""
        if self.image_interval and not self._interval_task:
            self._interval_task = asyncio.create_task(self._interval_processing_loop())
            self.logger.info(f"🔄 Started interval processing (every {self.image_interval}s)")

    def stop_interval(self):
        """Stop the interval processing task."""
        if self._interval_task:
            self._interval_task.cancel()
            self.logger.info("⏹️ Stopped interval processing")

    async def handle_interval(self):
        """Handle a single interval processing cycle."""
        if not self._current_frame:
            self.logger.debug("No current frame available for processing")
            return

        self.logger.debug(
            f"🔄 Running interval processing (frame: {self._current_frame.size if self._current_frame else None})"
        )

        # Process through pre-processors
        processed_data = {}
        for i, processor in enumerate(self.pre_processors):
            try:
                # Check if processor has async process method
                if hasattr(processor, "process") and asyncio.iscoroutinefunction(processor.process):
                    result = await processor.process(self._current_frame)
                else:
                    result = processor.process(self._current_frame)
                processed_data[f"processor_{i}_{type(processor).__name__}"] = result
                self.logger.debug(f"✅ Processed through {type(processor).__name__}")
            except Exception as e:
                self.logger.error(f"❌ Error in processor {type(processor).__name__}: {e}")
                processed_data[f"processor_{i}_{type(processor).__name__}_error"] = str(e)

        # Process through image processors if available
        for i, processor in enumerate(self.image_processors):
            try:
                if hasattr(processor, "process") and asyncio.iscoroutinefunction(processor.process):
                    result = await processor.process(self._current_frame)
                else:
                    result = processor.process(self._current_frame)
                processed_data[f"image_processor_{i}_{type(processor).__name__}"] = result
                self.logger.debug(f"✅ Processed through image processor {type(processor).__name__}")
            except Exception as e:
                self.logger.error(f"❌ Error in image processor {type(processor).__name__}: {e}")
                processed_data[f"image_processor_{i}_{type(processor).__name__}_error"] = str(e)

        # Send data to STS model if available
        if self.sts_model and hasattr(self.sts_model, "send_multimodal_data"):
            context_text = self._format_context(processed_data)
            await self.sts_model.send_multimodal_data(
                text=context_text, 
                image=self._current_frame, 
                data=processed_data
            )
            self.logger.debug("📤 Sent multimodal data to STS model")

        # Send data to regular model if available
        if self.model and processed_data:
            context_text = self._format_context(processed_data)
            # Note: This would need to be implemented based on your model's interface
            self.logger.debug("📤 Processed data available for model")


    def __init__(
        self,
        instructions: str,
        tools: Optional[List[Tool]] = None,
        pre_processors: Optional[List[PreProcessor]] = None,
        model: Optional[Model] = None,
        stt: Optional[STT] = None,
        tts: Optional[TTS] = None,
        vad: Optional[VAD] = None,
        turn_detection: Optional[TurnDetection] = None,
        sts_model: Optional[STS] = None,
        image_interval: Optional[int] = None,
        image_processors: Optional[List[ImageProcessor]] = None,
        video_transformer: Optional[VideoTransformer] = None,
        target_user_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the Agent.

        Args:
            instructions: System instructions for the agent
            tools: List of tools the agent can use
            pre_processors: List of pre-processors for input data
            model: AI model for generating responses
            stt: Speech-to-Text service
            tts: Text-to-Speech service
            vad: Voice Activity Detection service (optional)
            turn_detection: Turn detection service
            sts_model: Speech-to-Speech model (OpenAI Realtime API)
                      When provided, stt and tts are ignored
            image_interval: Interval in seconds for image processing (None to disable)
            image_processors: List of image processors to apply to video frames
            video_transformer: Video transformer to modify video frames before processing
            target_user_id: Specific user to capture video from (None for all users)
            bot_id: Unique bot ID (auto-generated if not provided)
            name: Display name for the bot
        """
        self.instructions = instructions
        self.tools = tools or []
        self.pre_processors = pre_processors or []
        self.model = model
        self.stt = stt
        self.tts = tts
        self.vad = vad
        self.turn_detection = turn_detection
        self.sts_model = sts_model
        self.image_interval = image_interval
        self.image_processors = image_processors or []
        self.video_transformer = video_transformer
        self.target_user_id = target_user_id
        self.bot_id = bot_id or f"agent-{uuid4()}"
        self.name = name or "AI Agent"

        self.validate_configuration()
        self.prepare_rtc()
        self.create_user() # TODO: always creating the user seems not ideal??

        # For STS + interval processing
        self._current_frame = None
        self._interval_task = None

        self._connection: Optional[rtc.RTCConnection] = None
        self._audio_track: Optional[audio_track.AudioStreamTrack] = None
        self._video_track: Optional[TransformedVideoTrack] = None
        self._is_running = False
        self._callback_executed = False

        self.logger = logging.getLogger(f"Agent[{self.bot_id}]")

    async def join(
        self,
        call,
    ) -> None:
        
        # 1. join the call (see if we need video)
        subscription_config = SubscriptionConfig(
            default=TrackSubscriptionConfig(
                track_types=[
                    TrackType.TRACK_TYPE_VIDEO,
                    TrackType.TRACK_TYPE_AUDIO,
                ]
            )
        )
        async with await rtc.join(
                call, self.bot_id, subscription_config=subscription_config
        ) as connection:
            self._connection = connection
            self._is_running = True

            self.logger.info(f"🤖 Agent joining call: {call.id}")
            # Set up audio track if available
            if self._audio_track:
                await connection.add_tracks(audio=self._audio_track)
                self.logger.info("🤖 Agent ready to speak")

            # Set up video track if available
            if self._video_track:
                await connection.add_tracks(video=self._video_track)
                self.logger.info("🎥 Agent ready to publish transformed video")

            # Start interval processing if configured
            if self.image_interval:
                self.start_interval()



                # Some callback to send the first message?
            self.llm.conversation_started()


            try:
                self.logger.info("🎧 Agent is active - press Ctrl+C to stop")
                await connection.wait()
            except Exception as e:
                self.logger.error(f"❌ Error during agent operation: {e}")
                self.logger.error(traceback.format_exc())
            finally:
                # Clean up interval processing
                self.stop_interval()
                self._is_running = False

    def validate_configuration(self):
        # Validate STS vs STT/TTS configuration
        if self.sts_model and (self.stt or self.tts):
            raise ValueError(
                "Cannot use both sts_model and stt/tts. "
                "STS (Speech-to-Speech) models handle both speech-to-text and text-to-speech internally."
            )

        if self.sts_model and self.model:
            self.logger.warning(
                "Using STS model with a separate model parameter. "
                "The STS model will handle conversation flow, and the model parameter will be ignored."
            )

    def prepare_rtc(self):
        # Set up audio track if TTS is available (traditional mode)
        if self.tts:
            self._audio_track = audio_track.AudioStreamTrack(framerate=16000)
            self.tts.set_output_track(self._audio_track)

        # Set up video track if video transformer is available
        if self.video_transformer:
            self._video_track = TransformedVideoTrack()
            self.logger.info("🎥 Video track initialized for transformation publishing")


    async def _interval_processing_loop(self):
        """Run interval processing loop at regular intervals."""
        while self._is_running:
            try:
                await asyncio.sleep(self.image_interval)

                if not self._is_running:
                    break

                await self.handle_interval()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in interval processing loop: {e}")
                self.logger.error(traceback.format_exc())

    def _format_context(self, processed_data: dict) -> str:
        """Format processed data into context text."""
        if not processed_data:
            return "No data available for analysis."

        context_parts = []
        for key, value in processed_data.items():
            if isinstance(value, dict):
                # Handle nested data structures
                formatted_value = ", ".join([f"{k}: {v}" for k, v in value.items()])
                context_parts.append(f"{key}: {{{formatted_value}}}")
            else:
                context_parts.append(f"{key}: {value}")

        return "Current analysis: " + "; ".join(context_parts)

    def create_user(self):
        pass