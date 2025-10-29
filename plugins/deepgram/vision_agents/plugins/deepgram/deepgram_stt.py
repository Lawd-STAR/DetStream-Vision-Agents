import asyncio
import logging
import os
from typing import Optional

import numpy as np
from deepgram import AsyncDeepgramClient
from getstream.video.rtc.track_util import PcmData

from vision_agents.core import stt
from vision_agents.core.stt import TranscriptResponse
from vision_agents.core.edge.types import Participant

logger = logging.getLogger(__name__)


class STT(stt.STT):
    """
    Deepgram Speech-to-Text implementation using Flux model.

    Deepgram provides highly accurate speech-to-text transcription with support
    for multiple languages and models. This implementation uses the Flux model
    via the v2 listen WebSocket API for real-time streaming transcription.

    The implementation maintains a persistent WebSocket connection to Deepgram,
    automatically resamples audio to 16kHz (recommended by Deepgram), and
    processes streaming results in real-time.

    Events:
        - transcript: Emitted when a final transcript is available.
        - partial_transcript: Emitted when a partial transcript is available.
        - error: Emitted when an error occurs during transcription.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "flux-general-en",
        language: Optional[str] = None,
        eot_threshold: Optional[float] = None,
        eager_eot_threshold: Optional[float] = None,
        client: Optional[AsyncDeepgramClient] = None,
    ):
        """
        Initialize Deepgram STT.

        Args:
            api_key: Deepgram API key. If not provided, will use DEEPGRAM_API_KEY env var.
            model: Model to use for transcription. Defaults to "flux-general-en".
            language: Language code (e.g., "en", "es"). If not provided, auto-detection is used.
            eot_threshold: End-of-turn threshold for determining when a turn is complete.
            eager_eot_threshold: Eager end-of-turn threshold for faster turn detection.
            client: Optional pre-configured AsyncDeepgramClient instance.
        """
        super().__init__(provider_name="deepgram")

        self._current_participant = None
        if not api_key:
            api_key = os.environ.get("DEEPGRAM_API_KEY")

        if client is not None:
            self.client = client
        else:
            # Initialize AsyncDeepgramClient with api_key as named parameter
            if api_key:
                self.client = AsyncDeepgramClient(api_key=api_key)
            else:
                self.client = AsyncDeepgramClient()

        self.model = model
        self.language = language
        self.eot_threshold = eot_threshold
        self.eager_eot_threshold = eager_eot_threshold
        self.connection = None
        self._connection_ready = asyncio.Event()
        self._connection_context = None

    async def start(self):
        """
        Start the Deepgram WebSocket connection and begin listening for transcripts.
        """
        if self.connection is not None:
            logger.warning("Deepgram connection already started")
            return

        logger.info("Starting Deepgram v2 listen connection")

        try:
            # Build connection parameters
            connect_params = {
                "model": self.model,
                "encoding": "linear16",
                "sample_rate": "16000",
            }
            
            # Add optional parameters if specified
            if self.eot_threshold is not None:
                connect_params["eot_threshold"] = str(self.eot_threshold)
            if self.eager_eot_threshold is not None:
                connect_params["eager_eot_threshold"] = str(self.eager_eot_threshold)
            
            # Connect to Deepgram v2 listen WebSocket with timeout
            self._connection_context = self.client.listen.v2.connect(**connect_params)
            
            # Add timeout for connection establishment
            self.connection = await asyncio.wait_for(
                self._connection_context.__aenter__(),
                timeout=10.0
            )

            # Register event handlers
            self.connection.on("message", self._on_message)
            self.connection.on("error", self._on_error)
            self.connection.on("close", self._on_close)

            # Start listening - this internally handles the message loop
            logger.info("Starting Deepgram listener")
            asyncio.create_task(self.connection.start_listening())

            # Mark connection as ready
            self._connection_ready.set()

            logger.info("Deepgram connection established and listening")

        except asyncio.TimeoutError:
            error_msg = "Timeout connecting to Deepgram. Check your API key and network connection."
            logger.error(error_msg)
            raise ConnectionError(error_msg)
        except Exception as e:
            logger.error(f"Failed to start Deepgram connection: {e}", exc_info=e)
            raise

    def _on_message(self, message):
        """
        Event handler for messages from Deepgram.
        
        Args:
            message: The message object from Deepgram
        """
        logger.debug(f"Received message: {message}")

        try:
            # Extract message data
            if not hasattr(message, "type"):
                return

            # Handle TurnInfo messages (v2 API)
            if message.type == "TurnInfo":
                # Run async handler in the event loop
                asyncio.create_task(self._handle_turn_info_message(message))

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=e)

    def _on_error(self, error):
        """
        Event handler for errors from Deepgram.
        
        Args:
            error: The error from Deepgram
        """
        logger.error(f"Deepgram WebSocket error: {error}")

    def _on_close(self, _):
        """
        Event handler for connection close.
        """
        logger.info("Deepgram WebSocket connection closed")
        self._connection_ready.clear()

    async def _handle_turn_info_message(self, message):
        """
        Handle a TurnInfo message from Deepgram v2 API.

        Args:
            message: The TurnInfo message object from Deepgram
        """
        # Extract transcript text
        transcript_text = getattr(message, "transcript", "").strip()
        
        if not transcript_text:
            return

        # Get event type to determine if final or partial
        # "StartOfTurn" and "Update" = partial, "EndOfTurn" = final
        event = getattr(message, "event", "")
        is_final = event == "EndOfTurn"

        # Get end of turn confidence
        end_of_turn_confidence = getattr(message, "end_of_turn_confidence", 0.0)

        # Calculate average confidence from words
        words = getattr(message, "words", [])
        if words:
            confidences = [w.confidence for w in words if hasattr(w, "confidence")]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        else:
            avg_confidence = 0.0

        # Get audio duration
        audio_window_end = getattr(message, "audio_window_end", 0.0)
        duration_ms = int(audio_window_end * 1000)

        # Build response metadata
        response_metadata = TranscriptResponse(
            confidence=avg_confidence,
            language=self.language or "auto",
            audio_duration_ms=duration_ms,
            model_name=self.model,
            other={
                "end_of_turn_confidence": end_of_turn_confidence,
                "turn_index": getattr(message, "turn_index", None),
                "event": event,
            }
        )

        # Use the participant from the most recent process_audio call
        participant = self._current_participant

        if is_final:
            # Final transcript (event == "EndOfTurn")
            logger.debug(
                "Received final transcript from Deepgram",
                extra={
                    "text_length": len(transcript_text),
                    "avg_confidence": avg_confidence,
                    "end_of_turn_confidence": end_of_turn_confidence,
                    "event": event,
                },
            )
            self._emit_transcript_event(
                transcript_text, participant, response_metadata
            )
        else:
            # Partial transcript (event == "StartOfTurn" or "Update")
            logger.debug(
                "Received partial transcript from Deepgram",
                extra={
                    "text_length": len(transcript_text),
                    "end_of_turn_confidence": end_of_turn_confidence,
                    "event": event,
                },
            )
            self._emit_partial_transcript_event(
                transcript_text, participant, response_metadata
            )

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ):
        """
        Process audio data through Deepgram for transcription.

        This method sends audio to the existing WebSocket connection. The connection
        is started automatically on first use. Audio is automatically resampled to 16kHz.

        Args:
            pcm_data: The PCM audio data to process.
            participant: Optional participant metadata (currently not used in streaming mode).
        """
        if self.closed:
            logger.warning("Deepgram STT is closed, ignoring audio")
            return

        # Ensure connection is established
        if self.connection is None:
            logger.info("Connection not started, starting now")
            await self.start()
            logger.info("start completed")

        # Wait for connection to be ready
        await self._connection_ready.wait()

        try:
            # Resample to 16kHz mono (recommended by Deepgram)
            resampled_pcm = pcm_data.resample(16_000, 1)

            # Convert int16 samples to bytes
            audio_bytes = resampled_pcm.samples.tobytes()

            logger.info(
                "Sending audio to Deepgram",
                extra={"audio_bytes": len(audio_bytes)},
            )
            self._current_participant = participant

            await self.connection.send_media(audio_bytes)

        except Exception as e:
            logger.error(
                "Error sending audio to Deepgram",
                exc_info=e,
            )
            raise

    async def close(self):
        """
        Close the Deepgram connection and clean up resources.
        """
        logger.info("Closing Deepgram STT")

        # Mark as closed first
        await super().close()

        # Close connection
        if self.connection and self._connection_context:
            try:
                await self.connection.send_control({"type": "CloseStream"})
                await self._connection_context.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing connection: {e}")
            finally:
                self.connection = None
                self._connection_context = None
                self._connection_ready.clear()
