import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import httpx
from getstream.video.rtc.track_util import PcmData, AudioFormat
import numpy as np
import onnxruntime as ort
from hatch.cli import self
from transformers import WhisperFeatureExtractor

from vision_agents.core.agents import Conversation
from vision_agents.core.agents.agents import default_agent_options
from vision_agents.core.edge.types import Participant

from vision_agents.core.turn_detection import (
    TurnDetector,
    TurnStartedEvent,
    TurnEndedEvent,
)

import logging

logger = logging.getLogger(__name__)

# Base directory for storing model files
SMART_TURN_ONNX_FILENAME = "smart-turn-v3.0.onnx"
SMART_TURN_ONNX_URL = (
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.0.onnx"
)

SILERO_ONNX_FILENAME = "silero_vad.onnx"
SILERO_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

# Audio processing constants
CHUNK = 512  # Samples per chunk for VAD processing
RATE = 16000  # Sample rate in Hz (16kHz)
MAX_SEGMENT_DURATION_SECONDS = 8 # Maximum duration in seconds for a single audio segment

@dataclass
class Silence:
    trailing_silence_chunks: int = 0
    speaking_chunks: int = 0

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
    - Vad expects 512 samples, webrtc will send 20ms (so roughly 304 packets at 16khz)
    """

    def __init__(
        self,
        vad_reset_interval_seconds: float = 5.0,
        speech_probability_threshold: float = 0.5,
        pre_speech_buffer_ms: int = 200,
        silence_duration_ms: int = 3000,
    ):
        """
        Initialize Smart Turn Detection.

        Args:
            vad_reset_interval_seconds: Reset VAD internal state every N seconds to prevent drift
            speech_probability_threshold: Minimum probability to consider audio as speech (0.0-1.0)
            pre_speech_buffer_ms: Duration in ms to buffer before speech detection trigger
            silence_duration_ms: Duration of trailing silence in ms before ending a turn
        """
        super().__init__()

        # Configuration parameters
        self._audio_buffer = PcmData(sample_rate=RATE, channels=1, format=AudioFormat.F32)
        self._silence = Silence()
        self.vad_reset_interval_seconds = vad_reset_interval_seconds
        self.speech_probability_threshold = speech_probability_threshold
        self.pre_speech_buffer_ms = pre_speech_buffer_ms
        self.silence_duration_ms = silence_duration_ms

        # TODO: this is not the most efficient data structure for a deque behaviour
        self._pre_speech_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._active_segment: Optional[PcmData] = None
        self._turn_in_progress = False
        self._trailing_silence_ms = 2000
        self._tail_silence_ms = 0.0

        # TODO: move this to the right place
        self.options = default_agent_options()

    async def start(self):
        # Ensure model directory exists
        os.makedirs(self.options.model_dir, exist_ok=True)

        # Prepare both models in parallel
        await asyncio.gather(
            self._prepare_smart_turn(),
            self._prepare_silero_vad(),
        )

    async def _prepare_smart_turn(self):
        path = os.path.join(self.options.model_dir, SMART_TURN_ONNX_FILENAME)
        await ensure_model(path, SMART_TURN_ONNX_URL)
        self._whisper_extractor = await asyncio.to_thread(
            WhisperFeatureExtractor, chunk_length=8
        )
        # Load ONNX session in thread pool to avoid blocking event loop
        self.smart_turn = await asyncio.to_thread(self._build_smart_turn_session)

    async def _prepare_silero_vad(self):
        path = os.path.join(self.options.model_dir, SILERO_ONNX_FILENAME)
        await ensure_model(path, SILERO_ONNX_URL)
        # Initialize VAD in thread pool to avoid blocking event loop
        self.vad = await asyncio.to_thread(
            SileroVAD, path, reset_interval_seconds=self.vad_reset_interval_seconds
        )

    async def process_audio(
        self,
        audio_data: PcmData,
        participant: Participant,
        conversation: Optional[Conversation],
    ) -> None:
        """
        Process audio does the following:
        - ensure 16khz and float 32 format
        - detect if something is speech
        - create segments while people are speaking
        - when it reaches enough silence or 8 seconds run it through smart turn to see if turn is completed

        See these examples:
        - https://github.com/pipecat-ai/smart-turn/blob/main/record_and_predict.py#L94
        - https://docs.pipecat.ai/server/utilities/smart-turn/smart-turn-overview#param-max-duration-secs
        - https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/audio/turn/smart_turn/local_smart_turn_v3.py

        The tricky bit is the 8 seconds. smart turn always want 8 seconds.
        - Do we share silence + the end (like it's shown in example record and predict?)
        - Or do we share historical + length of new segment to 8 seconds. (this seems better)
        """

        # ensure audio is in the right format
        audio_data = audio_data.resample(16000).to_float32()
        self._audio_buffer = self._audio_buffer.append(audio_data)

        # TODO: ensuring samples are 512 could be done in the base class

        if len(self._audio_buffer.samples) < 512:
            # too small to process
            return

        audio_chunks = list(self._audio_buffer.chunks(512))
        self._audio_buffer = PcmData(sample_rate=RATE, channels=1, format=AudioFormat.F32)
        self._audio_buffer.append(audio_chunks[-1]) # add back the last one
        # this ensures we handle the situation when audio data can't be divided by 512. ie 900

        # detect speech in small 512 chunks, gather to larger audio segments with speech
        for chunk in audio_chunks[:-1]:
            # predict if this segment has speech
            speech_probability = await self.vad.predict_speech(chunk.samples)
            is_speech = speech_probability > self.speech_probability_threshold
            logger.info("Processing chunk %d, is speech: %s %s", len(chunk.samples), speech_probability, is_speech)

            if self._active_segment is not None:
                # add to the segment
                logger.info("adding to segment")

                # TODO: it's python, append should work on the object, this is wrong
                self._active_segment = self._active_segment.append(chunk)

                if is_speech:
                    self._silence.speaking_chunks += 1
                    if self._silence.speaking_chunks > 3:
                        self._silence.trailing_silence_chunks = 0
                        self._silence.speaking_chunks = 0
                else:
                    self._silence.trailing_silence_chunks += 1

                # TODO: make this testable

                trailing_silence_ms = self._silence.trailing_silence_chunks * 512 / 16000 * 1000
                long_silence = trailing_silence_ms > self._trailing_silence_ms
                max_duration_reached = self._active_segment.duration_ms >= MAX_SEGMENT_DURATION_SECONDS * 1000
                logger.info("trailing_silence_ms = %s, self._active_segment.duration_ms %s", trailing_silence_ms, self._active_segment.duration_ms)

                if long_silence or max_duration_reached:
                    # expand to 8 seconds with either silence or historical
                    merged = PcmData(sample_rate=RATE, channels=1, format=AudioFormat.F32)
                    merged = merged.append(self._pre_speech_buffer)
                    merged = merged.append(self._active_segment)
                    merged = merged.tail(8, True, "start")
                    # see if we've completed the turn
                    prediction = await self._predict_turn_completed(
                        merged, participant
                    )
                    turn_ended = prediction > 0.5
                    if turn_ended:
                        self._emit_end_turn_event(TurnEndedEvent(participant=participant))
                        self._active_segment = None
                        self._silence = Silence()
                        # add the active segment to the speech buffer, so the next iteration can reuse it if needed
                        self._pre_speech_buffer = PcmData(sample_rate=RATE, channels=1, format=AudioFormat.F32)
                        self._pre_speech_buffer = self._pre_speech_buffer.append(merged)
                        self._pre_speech_buffer = self._pre_speech_buffer.tail(8)
            elif is_speech and self._active_segment is None:
                logger.info("starting new segment")
                self._emit_start_turn_event(TurnStartedEvent(participant=participant))
                # create a new segment
                self._active_segment = PcmData(sample_rate=RATE, channels=1, format=AudioFormat.F32)
                self._active_segment = self._active_segment.append(chunk)
                self._silence = Silence()
            else:
                logger.info("no active segment")
                # keep last n audio packets in speech buffer
                self._pre_speech_buffer=self._pre_speech_buffer.append(chunk)
                self._pre_speech_buffer=self._pre_speech_buffer.tail(8)


    async def _predict_turn_completed(
        self, pcm: PcmData, participant: Participant
    ) -> float:
        return await asyncio.to_thread(self._blocking_predict_turn_completed, pcm, participant)

    def _blocking_predict_turn_completed(
        self, pcm: PcmData, participant: Participant
    ) -> float:
        """
        Predict whether an audio segment is complete (turn ended) or incomplete.

        Args:
            pcm: PcmData containing audio samples

        Returns:
            - probability: Probability of completion (sigmoid output)
        """
        # Truncate to 8 seconds (keeping the end) or pad to 8 seconds
        audio_array = pcm.tail(8.0)

        # Process audio using Whisper's feature extractor
        inputs = self._whisper_extractor(
            audio_array.resample(16000).to_float32().samples,
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

    def _build_smart_turn_session(self):
        path = os.path.join(self.options.model_dir, SMART_TURN_ONNX_FILENAME)
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(path, sess_options=so)


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
        self.session = ort.InferenceSession(model_path, sess_options=opts)
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

    async def predict_speech(self, chunk_f32: np.ndarray) -> float:
        return await asyncio.to_thread(self._predict_speech, chunk_f32=chunk_f32)

    def _predict_speech(self, chunk_f32: np.ndarray) -> float:
        """
        Compute speech probability for one chunk of length 512 (float32, mono).
        Returns a scalar float.
        """
        # Ensure shape (1, 512) and concat context
        x = np.reshape(chunk_f32, (1, -1))
        if x.shape[1] != CHUNK:
            # Raise on incorrect usage
            raise ValueError("incorrect usage for predict speech. only send audio data in chunks of 512. got %d", x.shape[1])
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
            async with httpx.AsyncClient(
                timeout=300.0, follow_redirects=True
            ) as client:
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
