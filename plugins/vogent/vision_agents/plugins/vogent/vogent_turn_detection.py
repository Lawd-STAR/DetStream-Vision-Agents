import asyncio
import os
import time
from dataclasses import dataclass
from typing import Optional, Any

import httpx
import numpy as np
from faster_whisper import WhisperModel
from getstream.video.rtc.track_util import PcmData, AudioFormat
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

# Silero VAD model (reused from smart_turn)
SILERO_ONNX_FILENAME = "silero_vad.onnx"
SILERO_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

# Audio processing constants
CHUNK = 512  # Samples per chunk for VAD processing
RATE = 16000  # Sample rate in Hz (16kHz)


@dataclass
class Silence:
    trailing_silence_chunks: int = 0
    speaking_chunks: int = 0


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
    Blogpost: https://blog.vogent.ai/posts/voturn-80m-state-of-the-art-turn-detection-for-voice-agents
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
        model_dir: str = "/tmp/vogent_models",
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
            model_dir: Directory to store model files
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
        self.model_dir = model_dir
        
        # Audio buffering for processing
        self._audio_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._silence = Silence()
        self._pre_speech_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._active_segment: Optional[PcmData] = None
        self._trailing_silence_ms = self.silence_duration_ms
        
        # Producer-consumer pattern: audio packets go into buffer, background task processes them
        self._audio_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task[Any]] = None
        self._shutdown_event = asyncio.Event()
        
        # Model instances (initialized in start())
        self.vad = None
        self.whisper = None
        self.vogent = None

    async def start(self):
        """Initialize models and prepare for turn detection."""
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Prepare models in parallel
        await asyncio.gather(
            self._prepare_silero_vad(),
            self._prepare_whisper(),
            self._prepare_vogent(),
        )
        
        # Start background processing task
        self._processing_task = asyncio.create_task(self._process_audio_loop())
        
        # Call parent start method
        await super().start()

    async def _prepare_silero_vad(self) -> None:
        """Load Silero VAD model for speech detection."""
        path = os.path.join(self.model_dir, SILERO_ONNX_FILENAME)
        await ensure_model(path, SILERO_ONNX_URL)
        # Initialize VAD in thread pool to avoid blocking event loop
        self.vad = await asyncio.to_thread(
            lambda: SileroVAD(path, reset_interval_seconds=self.vad_reset_interval_seconds)  # type: ignore
        )

    async def _prepare_whisper(self) -> None:
        """Load faster-whisper model for transcription."""
        logger.info(f"Loading faster-whisper model: {self.whisper_model_size}")
        # Load whisper in thread pool to avoid blocking event loop
        self.whisper = await asyncio.to_thread(  # type: ignore[func-returns-value]
            lambda: WhisperModel(self.whisper_model_size, device="cpu", compute_type="int8")
        )
        logger.info("Faster-whisper model loaded")

    async def _prepare_vogent(self) -> None:
        """Load Vogent turn detection model."""
        logger.info("Loading Vogent turn detection model")
        # Load vogent in thread pool to avoid blocking event loop
        # Note: compile_model=False to avoid torch.compile issues with edge cases
        self.vogent = await asyncio.to_thread(  # type: ignore[func-returns-value]
            lambda: VogentDetector(
                compile_model=True,
                warmup=True,
                device=None,
                model_name="vogent/Vogent-Turn-80M",
                revision="main",
            )
        )
        logger.info("Vogent turn detection model loaded")

    async def process_audio(
        self,
        audio_data: PcmData,
        participant: Participant,
        conversation: Optional[Conversation],
    ) -> None:
        """
        Fast, non-blocking audio packet enqueueing.
        Actual processing happens in background task.
        """
        # Just enqueue the audio packet - fast and non-blocking
        await self._audio_queue.put((audio_data, participant, conversation))

    async def _process_audio_loop(self):
        """
        Background task that continuously processes audio from the queue.
        This is where the actual VAD and turn detection logic runs.
        """
        while not self._shutdown_event.is_set():
            try:
                # Wait for audio packet with timeout to allow shutdown
                audio_data, participant, conversation = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=1.0
                )

                # Process the audio packet
                await self._process_audio_packet(audio_data, participant, conversation)

            except asyncio.TimeoutError:
                # Timeout is expected - continue loop to check shutdown
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}")

    async def _process_audio_packet(
        self,
        audio_data: PcmData,
        participant: Participant,
        conversation: Optional[Conversation],
    ) -> None:
        """
        Process audio packet through VAD -> Whisper -> Vogent pipeline.
        
        This method:
        1. Buffers audio and processes in 512-sample chunks
        2. Uses VAD to detect speech
        3. Creates segments while people are speaking
        4. When reaching silence or max duration:
           - Transcribes segment with Whisper
           - Checks turn completion with Vogent (audio + text)
        
        Args:
            audio_data: PcmData object containing audio samples
            participant: Participant that's speaking
            conversation: Conversation history for context
        """
        # Ensure audio is in the right format: 16kHz, float32
        audio_data = audio_data.resample(RATE).to_float32()
        self._audio_buffer = self._audio_buffer.append(audio_data)

        if len(self._audio_buffer.samples) < CHUNK:
            # Too small to process
            return

        # Split into 512-sample chunks
        audio_chunks = list(self._audio_buffer.chunks(CHUNK))
        self._audio_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._audio_buffer.append(audio_chunks[-1])  # Add back the last one
        # This ensures we handle the situation when audio data can't be divided by 512

        # Detect speech in small 512 chunks, gather to larger audio segments with speech
        for chunk in audio_chunks[:-1]:
            # Predict if this segment has speech
            if self.vad is None:
                continue
                
            speech_probability = self.vad.predict_speech(chunk.samples)
            is_speech = speech_probability > self.speech_probability_threshold

            if self._active_segment is not None:
                # Add to the segment
                self._active_segment.append(chunk)

                if is_speech:
                    self._silence.speaking_chunks += 1
                    if self._silence.speaking_chunks > 3:
                        self._silence.trailing_silence_chunks = 0
                        self._silence.speaking_chunks = 0
                else:
                    self._silence.trailing_silence_chunks += 1

                trailing_silence_ms = (
                    self._silence.trailing_silence_chunks * CHUNK / RATE * 1000 * 5  # DTX correction
                )
                long_silence = trailing_silence_ms > self._trailing_silence_ms
                max_duration_reached = (
                    self._active_segment.duration_ms
                    >= self.max_segment_duration_seconds * 1000
                )

                if long_silence or max_duration_reached:
                    # Expand to 8 seconds with either silence or historical
                    merged = PcmData(
                        sample_rate=RATE, channels=1, format=AudioFormat.F32
                    )
                    merged.append(self._pre_speech_buffer)
                    merged.append(self._active_segment)
                    merged = merged.tail(8, True, "start")
                    
                    # Transcribe the segment with Whisper
                    transcription = await self._transcribe_segment(merged)
                    
                    # Get previous line from conversation for context
                    prev_line = self._get_previous_line(conversation)
                    
                    # Check if turn is complete using Vogent (multimodal: audio + text)
                    is_complete = await self._predict_turn_completed(
                        merged,
                        prev_line=prev_line,
                        curr_line=transcription,
                    )
                    
                    if is_complete:
                        self._emit_end_turn_event(
                            TurnEndedEvent(
                                participant=participant,
                                confidence=1.0,  # Vogent gives probability, we already thresholded
                                trailing_silence_ms=trailing_silence_ms,
                                duration_ms=self._active_segment.duration_ms,
                            )
                        )
                        self._active_segment = None
                        self._silence = Silence()
                        # Add the merged segment to the speech buffer for next iteration
                        self._pre_speech_buffer = PcmData(
                            sample_rate=RATE, channels=1, format=AudioFormat.F32
                        )
                        self._pre_speech_buffer.append(merged)
                        self._pre_speech_buffer = self._pre_speech_buffer.tail(8)
                        
            elif is_speech and self._active_segment is None:
                self._emit_start_turn_event(TurnStartedEvent(participant=participant))
                # Create a new segment
                self._active_segment = PcmData(
                    sample_rate=RATE, channels=1, format=AudioFormat.F32
                )
                self._active_segment.append(chunk)
                self._silence = Silence()
            else:
                # Keep last n audio packets in speech buffer
                self._pre_speech_buffer.append(chunk)
                self._pre_speech_buffer = self._pre_speech_buffer.tail(8)

    async def wait_for_processing_complete(self, timeout: float = 5.0) -> None:
        """Wait for all queued audio to be processed. Useful for testing."""
        start_time = time.time()
        while self._audio_queue.qsize() > 0 and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.01)

        # Give a small buffer for the processing to complete
        await asyncio.sleep(0.1)

    async def stop(self):
        """Stop turn detection and cleanup background task."""
        await super().stop()

        if self._processing_task:
            self._shutdown_event.set()
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None

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
        
        if self.whisper is None:
            return ""
        
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

        # Truncate to 8 seconds
        audio_array = pcm.tail(8, False).samples
        
        if self.vogent is None:
            return False
        
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


# TODO: maybe move to utility class
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
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self.context_size = 64  # Silero uses 64-sample context at 16 kHz
        self.reset_interval_seconds = reset_interval_seconds
        self._state: np.ndarray = np.zeros((2, 1, 128), dtype=np.float32)  # (2, B, 128)
        self._context: np.ndarray = np.zeros((1, 64), dtype=np.float32)
        self._init_states()

    def _init_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)  # (2, B, 128)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)
        self._last_reset_time = time.time()

    def maybe_reset(self):
        if (time.time() - self._last_reset_time) >= self.reset_interval_seconds:
            self._init_states()

    def predict_speech(self, chunk_f32: np.ndarray) -> float:
        """
        Compute speech probability for one chunk of length 512 (float32, mono).
        Returns a scalar float.
        """
        # Ensure shape (1, 512) and concat context
        x = np.reshape(chunk_f32, (1, -1))
        if x.shape[1] != CHUNK:
            # Raise on incorrect usage
            raise ValueError(
                f"incorrect usage for predict speech. only send audio data in chunks of 512. got {x.shape[1]}"
            )
        x = np.concatenate((self._context, x), axis=1)

        # Run ONNX
        ort_inputs = {
            "input": x.astype(np.float32),
            "state": self._state,
            "sr": np.array(16000, dtype=np.int64),
        }
        outputs = self.session.run(None, ort_inputs)
        out, self._state = outputs

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


