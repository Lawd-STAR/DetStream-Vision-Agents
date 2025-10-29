import abc
import logging
import uuid
from typing import Optional
from getstream.video.rtc.track_util import PcmData

from ..edge.types import Participant
from vision_agents.core.events.manager import EventManager
from . import events
from .events import TranscriptResponse

logger = logging.getLogger(__name__)


class STT(abc.ABC):
    """
    Abstract base class for Speech-to-Text implementations.

    Subclasses implement this and have to call
    - _emit_partial_transcript_event
    - _emit_transcript_event
    - _emit_error_event for temporary errors

    process_audio is currently called every 20ms. The integration with turn keeping could be improved
    """
    closed: bool = False

    def __init__(
        self,
        provider_name: Optional[str] = None,
    ):
        self.session_id = str(uuid.uuid4())
        self.provider_name = provider_name or self.__class__.__name__

        self.events = EventManager()
        self.events.register_events_from_module(events, ignore_not_compatible=True)

    def _emit_transcript_event(
        self,
        text: str,
        participant: Participant,
        response: TranscriptResponse,
    ):
        """
        Emit a final transcript event with structured data.

        Args:
            text: The transcribed text.
            participant: Participant metadata.
            response: Transcription response metadata.
        """
        self.events.send(events.STTTranscriptEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            text=text,
            participant=participant,
            response=response,
        ))

    def _emit_partial_transcript_event(
        self,
        text: str,
        participant: Participant,
        response: TranscriptResponse,
    ):
        """
        Emit a partial transcript event with structured data.

        Args:
            text: The partial transcribed text.
            participant: Participant metadata.
            response: Transcription response metadata.
        """
        self.events.send(events.STTPartialTranscriptEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            text=text,
            participant=participant,
            response=response,
        ))

    def _emit_error_event(
        self,
        error: Exception,
        context: str = "",
        participant: Optional[Participant] = None,
    ):
        """
        Emit an error event. Note this should only be emitted for temporary errors.
        Permanent errors due to config etc should be directly raised
        """
        self.events.send(events.STTErrorEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            error=error,
            context=context,
            participant=participant,
            error_code=getattr(error, "error_code", None),
            is_recoverable=not isinstance(error, (SystemExit, KeyboardInterrupt)),
        ))

    @abc.abstractmethod
    async def process_audio(
        self, pcm_data: PcmData, participant: Optional[Participant] = None,
    ):
        pass

    async def start(self):
        pass

    async def close(self):
        self.closed = True
