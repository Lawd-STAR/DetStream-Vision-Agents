import os
import time
import urllib
from typing import Optional

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

SMART_TURN_ONNX_PATH = "smart-turn-v3.0.onnx"
SMART_TURN_ONNX_URL = (
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.0.onnx"
)

SILERO_ONNX_PATH = "silero_vad.onnx"
SILERO_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

# Reset VAD internal state every N seconds
MODEL_RESET_STATES_TIME = 5.0
CHUNK = 512
RATE = 16000
VAD_THRESHOLD = 0.5  # speech probability threshold
PRE_SPEECH_MS = 200  # keep this many ms before trigger
STOP_MS = 1000  # end after this much trailing silence
MAX_DURATION_SECONDS = 8  # hard cap per segment


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
    """

    def __init__(self):
        super().__init__()
        # Use AudioSegmentCollector for automatic pre/post buffering and segment assembly
        self.collector = AudioSegmentCollector(
            pre_speech_ms=PRE_SPEECH_MS,
            post_speech_ms=STOP_MS,
            max_duration_s=MAX_DURATION_SECONDS,
            sample_rate=RATE,
            format=AudioFormat.F32,
        )
        self.turn_in_progress = False

    def start(self):
        # TODO: clean up the download functions
        # TODO: load session here
        ensure_model(SMART_TURN_ONNX_PATH, SMART_TURN_ONNX_URL)
        ensure_model(SILERO_ONNX_PATH, SILERO_ONNX_URL)

        self.vad = SileroVAD(SILERO_ONNX_PATH)

    async def _process_segment(self, pcm: PcmData, participant: Participant):
        segment_audio_f32 = pcm.samples
        dur_sec = segment_audio_f32.size / RATE
        print(f"Processing segment ({dur_sec:.2f}s)...")

        t0 = time.perf_counter()
        result = predict_endpoint(pcm)  # expects 16 kHz float32 mono
        dt_ms = (time.perf_counter() - t0) * 1000.0

        pred = result.get("prediction", 0)
        prob = result.get("probability", float("nan"))

        print("--------")
        print(f"Prediction: {'Complete' if pred == 1 else 'Incomplete'}")
        print(f"Probability of complete: {prob:.4f}")
        print(f"Inference time: {dt_ms:.2f} ms")

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
            is_speech = self.vad.prob(chunk.samples) > VAD_THRESHOLD

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
            await self._process_segment(segment, participant)

        # End turn if we're in a turn and collector is done (silence detected)
        if self.turn_in_progress and not self.collector.is_collecting:
            self._emit_end_turn_event(TurnEndedEvent(participant=participant))
            self.turn_in_progress = False
            print("Listening for speech...")


class SileroVAD:
    """Minimal Silero VAD ONNX wrapper for 16 kHz, mono, chunk=512."""

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.context_size = 64  # Silero uses 64-sample context at 16 kHz
        self._state = None
        self._context = None
        self._last_reset_time = time.time()
        self._init_states()

    def _init_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)  # (2, B, 128)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)

    def maybe_reset(self):
        if (time.time() - self._last_reset_time) >= MODEL_RESET_STATES_TIME:
            self._init_states()
            self._last_reset_time = time.time()

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


def ensure_model(path: str, url: str) -> str:
    # TODO: clean this up
    if not os.path.exists(path):
        print("Downloading Silero VAD ONNX model...")
        urllib.request.urlretrieve(url, path)
        print("ONNX model downloaded.")
    return path


def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
    # TODO: move to audio utils
    """Truncate audio to last n seconds or pad with zeros to meet n seconds."""
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    elif len(audio_array) < max_samples:
        # Pad with zeros at the beginning
        padding = max_samples - len(audio_array)
        return np.pad(audio_array, (padding, 0), mode="constant", constant_values=0)
    return audio_array


def build_session(onnx_path):
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so)


def predict_endpoint(pcm: PcmData):
    """
    Predict whether an audio segment is complete (turn ended) or incomplete.

    Args:
        pcm: PcmData containing audio samples

    Returns:
        Dictionary containing prediction results:
        - prediction: 1 for complete, 0 for incomplete
        - probability: Probability of completion (sigmoid output)
    """
    # Ensure it's 16khz and f32 format
    # Both resample and to_float32 are optimized to be no-ops if already in target format
    pcm = pcm.resample(16000).to_float32()

    feature_extractor = WhisperFeatureExtractor(chunk_length=8)
    session = build_session(SMART_TURN_ONNX_PATH)

    audio_array = pcm.samples
    # Truncate to 8 seconds (keeping the end) or pad to 8 seconds
    audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

    # Process audio using Whisper's feature extractor
    inputs = feature_extractor(
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
    outputs = session.run(None, {"input_features": input_features})

    # Extract probability (ONNX model returns sigmoid probabilities)
    probability = outputs[0][0].item()

    # Make prediction (1 for Complete, 0 for Incomplete)
    prediction = 1 if probability > 0.5 else 0

    return {
        "prediction": prediction,
        "probability": probability,
    }
