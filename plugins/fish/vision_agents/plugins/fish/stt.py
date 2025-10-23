import io
import logging
import os
import wave
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from fish_audio_sdk import Session, ASRRequest
from getstream.video.rtc.track_util import PcmData

from vision_agents.core import stt

if TYPE_CHECKING:
    from vision_agents.core.edge.types import Participant

logger = logging.getLogger(__name__)


class STT(stt.STT):
    """
    Fish Audio Speech-to-Text implementation.

    Fish Audio provides fast and accurate speech-to-text transcription with
    support for multiple languages and automatic language detection.

    This implementation operates in synchronous mode - it processes audio immediately
    and returns results to the base class, which then emits the appropriate events.

    Events:
        - transcript: Emitted when a complete transcript is available.
            Args: text (str), user_metadata (dict), metadata (dict)
        - error: Emitted when an error occurs during transcription.
            Args: error (Exception)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: Optional[str] = None,
        ignore_timestamps: bool = False,
        sample_rate: int = 16000,
        base_url: Optional[str] = None,
        client: Optional[Session] = None,
    ):
        """
        Initialize the Fish Audio STT service.

        Args:
            api_key: Fish Audio API key. If not provided, the FISH_API_KEY
                    environment variable will be used.
            language: Language code for transcription (e.g., "en", "zh"). If None,
                     automatic language detection will be used.
            ignore_timestamps: Skip timestamp processing for faster results.
            sample_rate: Sample rate of the audio in Hz (default: 16000).
            base_url: Optional custom API endpoint.
            client: Optionally pass in your own instance of the Fish Audio Session.
        """
        super().__init__(sample_rate=sample_rate, provider_name="fish")

        if not api_key:
            api_key = os.environ.get("FISH_API_KEY")

        if client is not None:
            self.client = client
        elif base_url:
            self.client = Session(api_key, base_url=base_url)
        else:
            self.client = Session(api_key)

        self.language = language
        self.ignore_timestamps = ignore_timestamps
        self._current_user: Optional[Union[Dict[str, Any], "Participant"]] = None

    def _pcm_to_wav_bytes(self, pcm_data: PcmData) -> bytes:
        """
        Convert PCM data to WAV format bytes.

        Args:
            pcm_data: PCM audio data from the audio pipeline.

        Returns:
            WAV format audio data as bytes.
        """
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            
            # Convert numpy array to bytes if needed
            if isinstance(pcm_data.samples, np.ndarray):
                wav_file.writeframes(pcm_data.samples.astype(np.int16).tobytes())
            else:
                wav_file.writeframes(pcm_data.samples)

        return wav_buffer.getvalue()

    async def _process_audio_impl(
        self,
        pcm_data: PcmData,
        user_metadata: Optional[Union[Dict[str, Any], "Participant"]] = None,
    ) -> Optional[List[Tuple[bool, str, Dict[str, Any]]]]:
        """
        Process audio data through Fish Audio for transcription.

        Fish Audio operates in synchronous mode - it processes audio immediately and
        returns results to the base class for event emission.

        Args:
            pcm_data: The PCM audio data to process.
            user_metadata: Additional metadata about the user or session.

        Returns:
            List of tuples (is_final, text, metadata) representing transcription results,
            or None if no results are available. Fish Audio returns final results only.
        """
        if self._is_closed:
            logger.warning("Fish Audio STT is closed, ignoring audio")
            return None

        # Store the current user context
        self._current_user = user_metadata

        # Check if we have valid audio data
        if not hasattr(pcm_data, "samples") or pcm_data.samples is None:
            logger.warning("No audio samples to process")
            return None

        # Check for empty audio
        if isinstance(pcm_data.samples, np.ndarray) and pcm_data.samples.size == 0:
            logger.debug("Received empty audio data")
            return None

        try:
            # Convert PCM to WAV format
            logger.debug(
                "Converting PCM to WAV",
                extra={"sample_rate": self.sample_rate},
            )
            wav_data = self._pcm_to_wav_bytes(pcm_data)

            # Build ASR request
            asr_request = ASRRequest(
                audio=wav_data,
                language=self.language,
                ignore_timestamps=self.ignore_timestamps,
            )

            # Send to Fish Audio API
            logger.debug(
                "Sending audio to Fish Audio ASR",
                extra={"audio_bytes": len(wav_data)},
            )
            response = self.client.asr(asr_request)

            # Extract transcript text
            transcript_text = response.text.strip()

            if not transcript_text:
                logger.debug("No transcript returned from Fish Audio")
                return None

            # Build metadata from response
            metadata: Dict[str, Any] = {
                "audio_duration_ms": response.duration,
                "language": self.language or "auto",
                "model_name": "fish-audio-asr",
            }

            # Include segments if timestamps were requested
            if not self.ignore_timestamps and response.segments:
                metadata["segments"] = [
                    {
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end,
                    }
                    for segment in response.segments
                ]

            logger.debug(
                "Received transcript from Fish Audio",
                extra={
                    "text_length": len(transcript_text),
                    "duration_ms": response.duration,
                },
            )

            # Return as final result (Fish Audio doesn't support streaming/partial results)
            return [(True, transcript_text, metadata)]

        except Exception as e:
            logger.error(
                "Error during Fish Audio transcription",
                exc_info=e,
            )
            # Let the base class handle error emission
            raise

    async def close(self):
        """Close the Fish Audio STT service and clean up resources."""
        if self._is_closed:
            logger.debug("Fish Audio STT service already closed")
            return

        logger.info("Closing Fish Audio STT service")
        await super().close()

