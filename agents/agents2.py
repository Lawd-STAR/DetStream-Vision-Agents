import asyncio
import logging
import traceback
from typing import Optional, Callable, List, Protocol, Any
from uuid import uuid4

from getstream.models import User
from getstream.video import rtc
from getstream.video.rtc import audio_track
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig, TrackType

from agents.agents import (
    TransformedVideoTrack,
    Tool,
    PreProcessor,
    TurnDetection,
    ImageProcessor,
    VideoTransformer,
    STT,
    TTS,
    VAD,
    STS, LLM
)

"""
TODO:
- improve naming on interval
- how to ping the LLM when connection is ready?
"""

class Processor(Protocol):
    def start(self, data: Any) -> Any:
        """Any initial setup"""
        ...

    def receive_audio(self):
        pass

    def receive_video(self):
        pass

    def state(self):
        """return state for the llm"""
        pass


class Agent:
    def __init__(
        self,


        # llm, optionally with sts capabilities
        llm: Optional[LLM] = None,

        # setup stt, tts, and turn detection if not using realtime/sts
        stt: Optional[STT] = None,
        tts: Optional[TTS] = None,
        turn_detection: Optional[TurnDetection] = None,

        # the agent's user info
        agent_user: Optional[User] = None,

        # for video agents. gather data at an interval
        # - roboflow/ yolo typically run continuously
        # - often combined with API calls to fetch stats etc
        # - state from each processor is passed to the LLM
        processors: Optional[List[PreProcessor]] = None,

        # transformers dont keep state/ aren't relevant for the LLM
        # just for applying sound or video effects
        video_transformer: Optional[VideoTransformer] = None,
    ):
        # Create agent user if not provided
        if agent_user is None:
            agent_id = f"agent-{uuid4()}"
            # Create a basic User object with required parameters
            self.agent_user = User(
                id=agent_id, 
                banned=False, 
                online=True, 
                role="user",
                custom={"name": "AI Agent"},
                teams_role={}
            )
        else:
            self.agent_user = agent_user

        self.logger = logging.getLogger(f"Agent[{self.agent_user.id}]")

        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.turn_detection = turn_detection
        self.processors = processors or []
        self.video_transformer = video_transformer

        # validation time
        self.validate_configuration()
        self.prepare_rtc()
        self.create_user()



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
                call, self.agent_user.id, subscription_config=subscription_config
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

            # Start interval processing
            for processor in self.processors:
                processor.start()

            # Some callback to send the first message?
            if self.llm and hasattr(self.llm, 'conversation_started'):
                self.llm.conversation_started(self.processors)


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
        pass

    def prepare_rtc(self):
        # Set up audio track if TTS is available (traditional mode)
        if self.tts:
            self._audio_track = audio_track.AudioStreamTrack(framerate=16000)
            self.tts.set_output_track(self._audio_track)

        # Set up video track if video transformer is available
        if self.video_transformer:
            self._video_track = TransformedVideoTrack()
            self.logger.info("🎥 Video track initialized for transformation publishing")

        # For STS + interval processing
        self._current_frame = None
        self._interval_task = None

        self._connection: Optional[rtc.RTCConnection] = None
        self._audio_track: Optional[audio_track.AudioStreamTrack] = None
        self._video_track: Optional[TransformedVideoTrack] = None
        self._is_running = False
        self._callback_executed = False


    def create_user(self):
        """Create user - placeholder for any user setup logic."""
        pass

    def stop_interval(self):
        """Stop any interval processing."""
        if hasattr(self, '_interval_task') and self._interval_task:
            self._interval_task.cancel()