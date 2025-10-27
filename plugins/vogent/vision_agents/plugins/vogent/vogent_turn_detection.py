import asyncio
import os
import time
from typing import Optional

import httpx
import numpy as np
from faster_whisper import WhisperModel
from getstream.video.rtc.track_util import AudioFormat, AudioSegmentCollector, PcmData
from vogent_turn import TurnDetector as VogentDetector

from vision_agents.core.agents import Conversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.turn_detection import (
    TurnDetector,
    TurnStartedEvent,
    TurnEndedEvent,
)

import logging

logger = logging.getLogger(__name__)

# Base directory for storing model files
MODEL_BASE_DIR = os.path.join(os.path.dirname(__file__), "models")

# Silero VAD model (reused from smart_turn)
SILERO_ONNX_FILENAME = "silero_vad.onnx"
SILERO_ONNX_PATH = os.path.join(MODEL_BASE_DIR, SILERO_ONNX_FILENAME)
SILERO_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

# Audio processing constants
CHUNK = 512  # Samples per chunk for VAD processing
RATE = 16000  # Sample rate in Hz (16kHz)


class VogentTurnDetection(TurnDetector):
    """
    Vogent Turn Detection combines audio intonation and text context for accurate turn detection.
    
    This implementation:
    1. Uses Silero VAD to detect when speech starts/stops
    2. Uses faster-whisper to transcribe audio in real-time
    3. Uses Vogent Turn model (multimodal) to detect turn completion
    
    Vogent operates on both audio features AND text context, making it more accurate
    than audio-only approaches, especially for handling incomplete thoughts.
    
    Reference: https://github.com/vogent/vogent-turn
    """

    def __init__(
        self,
        whisper_model_size: str = "tiny",
        vad_reset_interval_seconds: float = 5.0,
        speech_probability_threshold: float = 0.5,
        pre_speech_buffer_ms: int = 200,
        silence_duration_ms: int = 1000,
        max_segment_duration_seconds: int = 8,
        vogent_threshold: float = 0.5,
    ):
        """
        Initialize Vogent Turn Detection.
        
        Args:
            whisper_model_size: Faster-whisper model size (tiny, base, small, medium, large)
            vad_reset_interval_seconds: Reset VAD internal state every N seconds to prevent drift
            speech_probability_threshold: Minimum probability to consider audio as speech (0.0-1.0)
            pre_speech_buffer_ms: Duration in ms to buffer before speech detection trigger
            silence_duration_ms: Duration of trailing silence in ms before ending a turn
            max_segment_duration_seconds: Maximum duration in seconds for a single audio segment
            vogent_threshold: Threshold for vogent turn completion probability (0.0-1.0)
        """
        super().__init__()
        
        # Configuration parameters
        self.whisper_model_size = whisper_model_size
        self.vad_reset_interval_seconds = vad_reset_interval_seconds
        self.speech_probability_threshold = speech_probability_threshold
        self.pre_speech_buffer_ms = pre_speech_buffer_ms
        self.silence_duration_ms = silence_duration_ms
        self.max_segment_duration_seconds = max_segment_duration_seconds
        self.vogent_threshold = vogent_threshold
        
        # Use AudioSegmentCollector for automatic pre/post buffering and segment assembly
        self.collector = AudioSegmentCollector(
            pre_speech_ms=self.pre_speech_buffer_ms,
            post_speech_ms=self.silence_duration_ms,
            max_duration_s=self.max_segment_duration_seconds,
            sample_rate=RATE,
            format=AudioFormat.F32,
        )
        
        # Track turn state
        self.turn_in_progress = False
        self.current_transcription = ""
        self.previous_transcription = ""
        
        # Model instances (initialized in start())
        self.vad = None
        self.whisper = None
        self.vogent = None

    async def start(self):
        """Initialize models and prepare for turn detection."""
        # Ensure model directory exists
        os.makedirs(MODEL_BASE_DIR, exist_ok=True)
        
        # Prepare models in parallel
        await asyncio.gather(
            self._prepare_silero_vad(),
            self._prepare_whisper(),
            self._prepare_vogent(),
        )

    async def _prepare_silero_vad(self):
        """Load Silero VAD model for speech detection."""
        await ensure_model(SILERO_ONNX_PATH, SILERO_ONNX_URL)
        # Initialize VAD in thread pool to avoid blocking event loop
        self.vad = await asyncio.to_thread(
            SileroVAD, 
            SILERO_ONNX_PATH, 
            reset_interval_seconds=self.vad_reset_interval_seconds
        )

    async def _prepare_whisper(self):
        """Load faster-whisper model for transcription."""
        logger.info(f"Loading faster-whisper model: {self.whisper_model_size}")
        # Load whisper in thread pool to avoid blocking event loop
        self.whisper = await asyncio.to_thread(
            WhisperModel,
            self.whisper_model_size,
            device="cpu",
            compute_type="int8",
        )
        logger.info("Faster-whisper model loaded")

    async def _prepare_vogent(self):
        """Load Vogent turn detection model."""
        logger.info("Loading Vogent turn detection model")
        # Load vogent in thread pool to avoid blocking event loop
        # Note: compile_model=False to avoid torch.compile issues with edge cases
        self.vogent = await asyncio.to_thread(
            VogentDetector,
            compile_model=False,
            warmup=False,
        )
        logger.info("Vogent turn detection model loaded")

    async def process_audio(
        self,
        audio_data: PcmData,
        participant: Participant,
        conversation: Optional[Conversation],
    ) -> None:
        """
        Process audio and detect turn completion using multimodal approach.
        
        Args:
            audio_data: PcmData object containing audio samples
            participant: Participant that's speaking
            conversation: Conversation history for context
        """
        # Ensure audio is in the right format: 16kHz, float32
        pcm = audio_data.resample(RATE).to_float32()

        # Track segments for final turn-end processing
        segments_to_process = []

        # Process audio in 512-sample chunks for VAD
        for chunk in pcm.chunks(chunk_size=CHUNK):
            # Run VAD on the chunk
            is_speech = self.vad.prob(chunk.samples) > self.speech_probability_threshold

            # Emit turn start on first speech detection
            if is_speech and not self.turn_in_progress:
                self._emit_start_turn_event(TurnStartedEvent(participant=participant))
                self.turn_in_progress = True
                self.current_transcription = ""

            # Feed chunk to collector, which handles pre/post buffering
            segment = self.collector.add_chunk(chunk, is_speech=is_speech)

            # Collect segments for processing
            if segment is not None:
                segments_to_process.append(segment)

        # Process all collected segments
        for segment in segments_to_process:
            # Transcribe the segment
            transcription = await self._transcribe_segment(segment)
            self.current_transcription = transcription
            
            # Only check turn completion if we have transcription
            if self.current_transcription.strip():
                # Get previous line from conversation
                prev_line = self._get_previous_line(conversation)
                
                # Check if turn is complete using vogent
                is_complete = await self._predict_turn_completed(
                    segment, 
                    prev_line=prev_line,
                    curr_line=self.current_transcription,
                )
                
                logger.debug(
                    f"Vogent prediction: is_complete={is_complete}, "
                    f"prev='{prev_line}', curr='{self.current_transcription}'"
                )

        # End turn if we're in a turn and collector is done (silence detected)
        if self.turn_in_progress and not self.collector.is_collecting:
            # Final check with vogent before ending turn
            if self.current_transcription and self.current_transcription.strip():
                prev_line = self._get_previous_line(conversation)
                # Get last segment for final prediction
                last_segment = segments_to_process[-1] if segments_to_process else None
                
                if last_segment is not None:
                    is_complete = await self._predict_turn_completed(
                        last_segment,
                        prev_line=prev_line,
                        curr_line=self.current_transcription,
                    )
                    
                    if is_complete:
                        self._emit_end_turn_event(TurnEndedEvent(participant=participant))
                        self.turn_in_progress = False
                        # Save current as previous for next turn
                        self.previous_transcription = self.current_transcription
                        self.current_transcription = ""
                else:
                    # No segment to check, end turn anyway due to silence
                    self._emit_end_turn_event(TurnEndedEvent(participant=participant))
                    self.turn_in_progress = False
                    self.previous_transcription = self.current_transcription
                    self.current_transcription = ""
            else:
                # No transcription, just end the turn
                self._emit_end_turn_event(TurnEndedEvent(participant=participant))
                self.turn_in_progress = False

    async def _transcribe_segment(self, pcm: PcmData) -> str:
        """
        Transcribe audio segment using faster-whisper.
        
        Args:
            pcm: PcmData containing audio samples
            
        Returns:
            Transcribed text
        """
        # Ensure it's 16khz and f32 format
        pcm = pcm.resample(16000).to_float32()
        audio_array = pcm.samples
        
        # Run transcription in thread pool to avoid blocking
        segments, info = await asyncio.to_thread(
            self.whisper.transcribe,
            audio_array,
            language="en",
            beam_size=1,
            vad_filter=False,  # We already did VAD
        )
        
        # Collect all text segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        
        transcription = " ".join(text_parts).strip()
        return transcription

    async def _predict_turn_completed(
        self, 
        pcm: PcmData, 
        prev_line: str,
        curr_line: str,
    ) -> bool:
        """
        Predict whether the current turn is complete using Vogent.
        
        Args:
            pcm: PcmData containing audio samples
            prev_line: Previous speaker's text (for context)
            curr_line: Current speaker's text
            
        Returns:
            True if turn is complete, False otherwise
        """
        # Ensure it's 16khz and f32 format
        pcm = pcm.resample(16000).to_float32()
        audio_array = pcm.samples
        
        # Truncate to 8 seconds (vogent requirement)
        audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)
        
        # Run vogent prediction in thread pool
        result = await asyncio.to_thread(
            self.vogent.predict,
            audio_array,
            prev_line=prev_line,
            curr_line=curr_line,
            sample_rate=16000,
            return_probs=True,
        )
        
        # Check if probability exceeds threshold
        is_complete = result['prob_endpoint'] > self.vogent_threshold
        logger.debug(
            f"Vogent probability: {result['prob_endpoint']:.3f}, "
            f"threshold: {self.vogent_threshold}, is_complete: {is_complete}"
        )
        
        return is_complete

    def _get_previous_line(self, conversation: Optional[Conversation]) -> str:
        """
        Extract the previous speaker's line from conversation history.
        
        Args:
            conversation: Conversation object with message history
            
        Returns:
            Previous line text, or empty string if not available
        """
        if conversation is None or not conversation.messages:
            return ""
        
        # Get the last message that's not from the current speaker
        # Typically this would be the assistant or another user
        for message in reversed(conversation.messages):
            if message.content and message.content.strip():
                # Remove terminal punctuation for better vogent performance
                text = message.content.strip().rstrip('.!?')
                return text
        
        return ""


class SileroVAD:
    """
    Minimal Silero VAD ONNX wrapper for 16 kHz, mono, chunk=512.
    
    Reused from smart_turn implementation.
    """

    def __init__(self, model_path: str, reset_interval_seconds: float = 5.0):
        """
        Initialize Silero VAD.
        
        Args:
            model_path: Path to the ONNX model file
            reset_interval_seconds: Reset internal state every N seconds to prevent drift
        """
        import onnxruntime as ort
        
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.context_size = 64  # Silero uses 64-sample context at 16 kHz
        self.reset_interval_seconds = reset_interval_seconds
        self._state = None
        self._context = None
        self._init_states()

    def _init_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)  # (2, B, 128)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)
        self._last_reset_time = time.time()

    def maybe_reset(self):
        if (time.time() - self._last_reset_time) >= self.reset_interval_seconds:
            self._init_states()

    def prob(self, chunk_f32: np.ndarray) -> float:
        """
        Compute speech probability for one chunk of length 512 (float32, mono).
        Returns a scalar float.
        """
        # Ensure shape (1, 512) and concat context
        x = np.reshape(chunk_f32, (1, -1))
        if x.shape[1] != CHUNK:
            # Return 0.0 for incomplete chunks instead of raising
            return 0.0
        x = np.concatenate((self._context, x), axis=1)

        # Run ONNX
        ort_inputs = {
            "input": x.astype(np.float32),
            "state": self._state,
            "sr": np.array(16000, dtype=np.int64),
        }
        out, self._state = self.session.run(None, ort_inputs)

        # Update context (keep last 64 samples)
        self._context = x[:, -self.context_size :]
        self.maybe_reset()

        # out shape is (1, 1) -> return scalar
        return float(out[0][0])


async def ensure_model(path: str, url: str) -> str:
    """
    Download a model file asynchronously if it doesn't exist.
    
    Args:
        path: Local path where the model should be saved
        url: URL to download the model from
        
    Returns:
        The path to the model file
    """
    if not os.path.exists(path):
        model_name = os.path.basename(path)
        logger.info(f"Downloading {model_name}...")
        
        try:
            async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    
                    # Write file in chunks to avoid loading entire file in memory
                    chunks = []
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        chunks.append(chunk)
                    
                    # Write all chunks to file in thread to avoid blocking event loop
                    def write_file():
                        with open(path, "wb") as f:
                            for chunk in chunks:
                                f.write(chunk)
                    
                    await asyncio.to_thread(write_file)
            
            logger.info(f"{model_name} downloaded.")
        except httpx.HTTPError as e:
            # Clean up partial download on error
            if os.path.exists(path):
                os.remove(path)
            raise RuntimeError(f"Failed to download {model_name}: {e}")
    
    return path


def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
    """Truncate audio to last n seconds or pad with zeros to meet n seconds."""
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    elif len(audio_array) < max_samples:
        # Pad with zeros at the beginning
        padding = max_samples - len(audio_array)
        return np.pad(audio_array, (padding, 0), mode="constant", constant_values=0)
    return audio_array

