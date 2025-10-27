import asyncio
import os
import time
from typing import Optional

import httpx
from getstream.video.rtc.track_util import PcmData, AudioSegmentCollector, AudioFormat
import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

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

SMART_TURN_ONNX_FILENAME = "smart-turn-v3.0.onnx"
SMART_TURN_ONNX_PATH = os.path.join(MODEL_BASE_DIR, SMART_TURN_ONNX_FILENAME)
SMART_TURN_ONNX_URL = (
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.0.onnx"
)

SILERO_ONNX_FILENAME = "silero_vad.onnx"
SILERO_ONNX_PATH = os.path.join(MODEL_BASE_DIR, SILERO_ONNX_FILENAME)
SILERO_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

# Audio processing constants
CHUNK = 512  # Samples per chunk for VAD processing
RATE = 16000  # Sample rate in Hz (16kHz)


class SmartTurnDetection(TurnDetector):
    """
    Daily's pipecat project did a really nice job with turn detection
    This package implements smart turn v3 as documented here
    https://github.com/pipecat-ai/smart-turn/tree/main

    It's based on a Whisper Tiny encoder and only look at audio features.
    This is only audio based, it doesn't understand what's said like the Vogent model.

    Due to this approach it's much faster.
    https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/

    A few things to keep in mind while working on this
    - It runs Silero VAD in front of it to ensure it only runs when the user is speaking.
    - Silero VAD uses 512 chunks, 16khz, 32 float encoded audio
    - Smart turn uses 16khz, 32 float encoded audio
    - Smart turn evaluates audio in 8s chunks. prefixed with silence at the beginning, but always 8s
    """

    def __init__(
            self,
            vad_reset_interval_seconds: float = 5.0,
            speech_probability_threshold: float = 0.5,
            pre_speech_buffer_ms: int = 200,
            silence_duration_ms: int = 1000,
            max_segment_duration_seconds: int = 8,
    ):
        """
        Initialize Smart Turn Detection.

        Args:
            vad_reset_interval_seconds: Reset VAD internal state every N seconds to prevent drift
            speech_probability_threshold: Minimum probability to consider audio as speech (0.0-1.0)
            pre_speech_buffer_ms: Duration in ms to buffer before speech detection trigger
            silence_duration_ms: Duration of trailing silence in ms before ending a turn
            max_segment_duration_seconds: Maximum duration in seconds for a single audio segment
        """
        super().__init__()

        # Configuration parameters
        self.vad_reset_interval_seconds = vad_reset_interval_seconds
        self.speech_probability_threshold = speech_probability_threshold
        self.pre_speech_buffer_ms = pre_speech_buffer_ms
        self.silence_duration_ms = silence_duration_ms
        self.max_segment_duration_seconds = max_segment_duration_seconds

        # Use AudioSegmentCollector for automatic pre/post buffering and segment assembly
        self.collector = AudioSegmentCollector(
            pre_speech_ms=self.pre_speech_buffer_ms,
            post_speech_ms=self.silence_duration_ms,
            max_duration_s=self.max_segment_duration_seconds,
            sample_rate=RATE,
            format=AudioFormat.F32,
        )
        self.turn_in_progress = False

    async def start(self):
        # Ensure model directory exists
        os.makedirs(MODEL_BASE_DIR, exist_ok=True)

        # Prepare both models in parallel
        await asyncio.gather(
            self._prepare_smart_turn(),
            self._prepare_silero_vad(),
        )

    async def _prepare_smart_turn(self):
        await ensure_model(SMART_TURN_ONNX_PATH, SMART_TURN_ONNX_URL)
        self._whisper_extractor = await asyncio.to_thread(WhisperFeatureExtractor, chunk_length=8)
        # Load ONNX session in thread pool to avoid blocking event loop
        self.smart_turn = await asyncio.to_thread(build_session, SMART_TURN_ONNX_PATH)

    async def _prepare_silero_vad(self):
        await ensure_model(SILERO_ONNX_PATH, SILERO_ONNX_URL)
        # Initialize VAD in thread pool to avoid blocking event loop
        self.vad = await asyncio.to_thread(
            SileroVAD,
            SILERO_ONNX_PATH,
            reset_interval_seconds=self.vad_reset_interval_seconds
        )

    async def process_audio(
            self,
            audio_data: PcmData,
            participant: Participant,
            conversation: Optional[Conversation],
    ) -> None:
        # Ensure audio is in the right format: 16kHz, float32
        # Both resample and to_float32 are optimized to be no-ops if already in target format
        pcm = audio_data.resample(RATE).to_float32()

        # Track segments for final turn-end processing
        segments_to_process = []

        # Process audio in 512-sample chunks
        for chunk in pcm.chunks(chunk_size=CHUNK):
            # Run VAD on the chunk
            is_speech = self.vad.prob(chunk.samples) > self.speech_probability_threshold

            # Emit turn start on first speech detection
            if is_speech and not self.turn_in_progress:
                self._emit_start_turn_event(TurnStartedEvent(participant=participant))
                self.turn_in_progress = True

            # Feed chunk to collector, which handles pre/post buffering
            segment = self.collector.add_chunk(chunk, is_speech=is_speech)

            # Collect segments for processing
            if segment is not None:
                segments_to_process.append(segment)

        # Process all collected segments
        for segment in segments_to_process:
            prediction = await self._predict_turn_completed(segment, participant)
            turn_completed = prediction > 0.5

        # End turn if we're in a turn and collector is done (silence detected)
        if self.turn_in_progress and not self.collector.is_collecting:
            self._emit_end_turn_event(TurnEndedEvent(participant=participant))
            self.turn_in_progress = False

    async def _predict_turn_completed(self, pcm: PcmData, participant: Participant) -> float:
        """
        Predict whether an audio segment is complete (turn ended) or incomplete.

        Args:
            pcm: PcmData containing audio samples

        Returns:
            - probability: Probability of completion (sigmoid output)
        """
        # Ensure it's 16khz and f32 format
        # Both resample and to_float32 are optimized to be no-ops if already in target format
        pcm = pcm.resample(16000).to_float32()

        # TODO: can we only init this once?

        audio_array = pcm.samples
        # Truncate to 8 seconds (keeping the end) or pad to 8 seconds
        audio_array = pcm.tail(8.0, True, "start")

        # Process audio using Whisper's feature extractor
        inputs = self._whisper_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="np",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True,
        )

        # Extract features and ensure correct shape for ONNX
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)  # Add batch dimension

        # Run ONNX inference
        outputs = self.smart_turn.run(None, {"input_features": input_features})

        # Extract probability (ONNX model returns sigmoid probabilities)
        probability = outputs[0][0].item()

        return probability


class SileroVAD:
    """Minimal Silero VAD ONNX wrapper for 16 kHz, mono, chunk=512."""

    def __init__(self, model_path: str, reset_interval_seconds: float = 5.0):
        """
        Initialize Silero VAD.

        Args:
            model_path: Path to the ONNX model file
            reset_interval_seconds: Reset internal state every N seconds to prevent drift
        """
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
        self._context = x[:, -self.context_size:]
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
                    # Use asyncio.to_thread for blocking file I/O operations
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


def build_session(onnx_path):
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so)

