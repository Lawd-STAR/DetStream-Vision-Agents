import asyncio
import logging
import traceback
from typing import Optional, Callable, List
from uuid import uuid4

from getstream.models import User
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

    def __init__(
        self,
        # llm, optionally with sts capabilities
        llm: Optional[LLM] = None,

        # setup stt, tts, and turn detection if not using realtime/sts
        stt: Optional[STT] = None,
        tts: Optional[TTS] = None,
        turn_detection: Optional[TurnDetection] = None,

        # the agent's name
        agent_user : Optional[User] = None,

        # for video agents. gather data at an interval
        # - roboflow/ yolo typically run continuously
        # - often combined with API calls to fetch stats etc.
        # - context from each processor is passed to the LLM
        processors: Optional[List[PreProcessor]] = None,

        # transformers dont keep state/ aren't relevant for the LLM
        # just for applying sound or video effects
        video_transformer: Optional[VideoTransformer] = None,
    ):
        self.logger = logging.getLogger(f"Agent[{self.agent_user.id}]")

        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.turn_detection = turn_detection
        self.agent_user = agent_user
        self.processors = processors
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
            self.llm.conversation_started(self.pre_processors)


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
        pass