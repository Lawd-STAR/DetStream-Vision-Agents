import logging
import os
from typing import AsyncIterator, Optional

from fish_audio_sdk import Session, TTSRequest
from getstream.video.rtc.audio_track import AudioStreamTrack
from vision_agents.core import tts

logger = logging.getLogger(__name__)


class TTS(tts.TTS):
    """
    Fish Audio Text-to-Speech implementation.
    
    Fish Audio provides high-quality, multilingual text-to-speech synthesis with
    support for voice cloning via reference audio.


    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        reference_id: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[Session] = None,
    ):
        """
        Initialize the Fish Audio TTS service.

        Args:
            api_key: Fish Audio API key. If not provided, the FISH_AUDIO_API_KEY
                    environment variable will be used.
            reference_id: Optional reference voice ID to use for synthesis.
            base_url: Optional custom API endpoint.
            client: Optionally pass in your own instance of the Fish Audio Session.
        """
        super().__init__(provider_name="fish")

        if not api_key:
            api_key = os.environ.get("FISH_API_KEY")

        if client is not None:
            self.client = client
        elif base_url:
            self.client = Session(api_key, base_url=base_url)
        else:
            self.client = Session(api_key)

        self.reference_id = reference_id

        # Fish Audio typically outputs at 44100 Hz, but we'll use 16000 for compatibility
        # Note: You may need to adjust this based on Fish Audio's actual output
        self.output_framerate = 16000

    def get_required_framerate(self) -> int:
        """Get the required framerate for Fish Audio TTS."""
        return self.output_framerate

    def get_required_stereo(self) -> bool:
        """Get whether Fish Audio TTS requires stereo audio."""
        return False  # Fish Audio typically returns mono audio

    def set_output_track(self, track: AudioStreamTrack) -> None:
        """
        Set the output audio track.
        
        Args:
            track: The audio track to output to.
        
        Raises:
            TypeError: If the track framerate doesn't match requirements.
        """
        if track.framerate != self.output_framerate:
            raise TypeError(
                f"Invalid framerate, audio track only supports {self.output_framerate}"
            )
        super().set_output_track(track)

    async def stream_audio(self, text: str, *_, **kwargs) -> AsyncIterator[bytes]:
        """
        Convert text to speech using Fish Audio API.

        Args:
            text: The text to convert to speech.
            **kwargs: Additional arguments to pass to TTSRequest (e.g., references).

        Returns:
            An async iterator of audio chunks as bytes.
        """
        # Build the TTS request
        tts_request_kwargs = {"text": text}
        
        # Add reference_id if configured
        if self.reference_id:
            tts_request_kwargs["reference_id"] = self.reference_id
        
        # Allow overriding via kwargs (e.g., for dynamic reference audio)
        tts_request_kwargs.update(kwargs)
        
        tts_request = TTSRequest(format="pcm", sample_rate=16000, normalize=True,reference_id="03397b4c4be74759b72533b663fbd001", **tts_request_kwargs)

        # Stream audio from Fish Audio
        audio_stream = self.client.tts.awaitable(tts_request)

        return audio_stream

    async def stop_audio(self) -> None:
        """
        Clears the queue and stops playing audio.
        
        This method can be used manually or under the hood in response to turn events.

        Returns:
            None
        """
        if self.track is not None:
            try:
                await self.track.flush()
                logger.info("🎤 Stopping audio track for Fish Audio TTS")
            except Exception as e:
                logger.error(f"Error flushing audio track: {e}")
        else:
            logger.warning("No audio track to stop")

