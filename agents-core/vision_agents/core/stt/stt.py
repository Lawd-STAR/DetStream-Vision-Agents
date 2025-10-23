import abc
import logging
import uuid
from typing import Optional, Dict, Any, Union
from getstream.video.rtc.track_util import PcmData

from ..edge.types import Participant
from vision_agents.core.events.manager import EventManager
from . import events

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
        user_metadata: Optional[Union[Dict[str, Any], Participant]],
        metadata: Dict[str, Any],
    ):
        """
        Emit a final transcript event with structured data.

        Args:
            text: The transcribed text.
            user_metadata: User-specific metadata.
            metadata: Transcription metadata (processing time, confidence, etc.).
        """
        self.events.send(events.STTTranscriptEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            text=text,
            user_metadata=user_metadata,
            confidence=metadata.get("confidence"),
            language=metadata.get("language"),
            processing_time_ms=metadata.get("processing_time_ms"),
            audio_duration_ms=metadata.get("audio_duration_ms"),
            model_name=metadata.get("model_name"),
            words=metadata.get("words"),
        ))

    def _emit_partial_transcript_event(
        self,
        text: str,
        user_metadata: Optional[Union[Dict[str, Any], Participant]],
        metadata: Dict[str, Any],
    ):
        """
        Emit a partial transcript event with structured data.

        Args:
            text: The partial transcribed text.
            user_metadata: User-specific metadata.
            metadata: Transcription metadata (processing time, confidence, etc.).
        """
        self.events.send(events.STTPartialTranscriptEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            text=text,
            user_metadata=user_metadata,
            confidence=metadata.get("confidence"),
            language=metadata.get("language"),
            processing_time_ms=metadata.get("processing_time_ms"),
            audio_duration_ms=metadata.get("audio_duration_ms"),
            model_name=metadata.get("model_name"),
            words=metadata.get("words"),
        ))

    def _emit_error_event(
        self,
        error: Exception,
        context: str = "",
        user_metadata: Optional[Union[Dict[str, Any], Participant]] = None,
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
            user_metadata=user_metadata,
            error_code=getattr(error, "error_code", None),
            is_recoverable=not isinstance(error, (SystemExit, KeyboardInterrupt)),
        ))

    @abc.abstractmethod
    async def process_audio(
        self, pcm_data: PcmData, participant: Optional[Participant] = None,
    ):
        pass

    async def close(self):
        self.closed = True
