import logging

import torch
import numpy as np
import warnings
import time
from typing import Dict, Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from vision_agents.core.edge.types import Participant
from vision_agents.core import vad
from vision_agents.core.vad.events import VADSpeechStartEvent

from getstream.video.rtc.track_util import PcmData

from vision_agents.core.events import AudioFormat


try:
    import onnxruntime as ort

    has_onnx = True
except ImportError:
    has_onnx = False


logger = logging.getLogger(__name__)


class VAD(vad.VAD):
    """
    Voice Activity Detection implementation using Silero VAD model.

    This class implements the VAD interface using the Silero VAD model,
    which is a high-performance speech detection model.

    Features:
    - Asymmetric thresholds for speech detection (activation_th and deactivation_th)
    - Automatic resampling to model's required rate (typically 16kHz)
    - GPU acceleration support with automatic fallback to CPU
    - Optional ONNX runtime support for potential performance improvements
    - Early partial events for real-time UI feedback during speech
    - Memory-efficient audio buffering using bytearrays
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        frame_size: Optional[int] = None,
        activation_th: float = 0.4,
        deactivation_th: float = 0.2,
        speech_pad_ms: int = 300,
        min_speech_ms: int = 250,
        max_speech_ms: int = 30000,
        model_rate: int = 16000,
        window_samples: int = 512,
        device: str = "cpu",
        partial_frames: int = 10,
        use_onnx: bool = False,
    ):
        """
        Initialize the Silero VAD.

        Args:
            sample_rate: Audio sample rate in Hz expected for input
            frame_size: (Deprecated) Size of audio frames to process, use window_samples instead
            activation_th: Threshold for starting speech detection (0.0 to 1.0)
            deactivation_th: Threshold for ending speech detection (0.0 to 1.0) (defaults to 0.7*activation_th)
            speech_pad_ms: Number of milliseconds to pad before/after speech
            min_speech_ms: Minimum milliseconds of speech to emit
            max_speech_ms: Maximum milliseconds of speech before forced flush
            model_rate: Sample rate the model operates on (typically 16000 Hz)
            window_samples: Number of samples per window (must be 512 for 16kHz, 256 for 8kHz)
            device: Device to run the model on ("cpu", "cuda", "cuda:0", etc.)
            partial_frames: Number of frames to process before emitting a "partial" event
            use_onnx: Whether to use ONNX runtime for inference instead of PyTorch
        """
        # Issue deprecation warning for frame_size
        if frame_size is not None:
            warnings.warn(
                "The 'frame_size' parameter is deprecated and will be removed in a future version. "
                "Use 'window_samples' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            window_samples = frame_size

        # Base VAD expects model spec: sample_rate, window size, channels, format
        super().__init__(
            sample_rate=model_rate,
            window_samples=window_samples,
            channels=1,
            audio_format=AudioFormat.PCM_S16,
            activation_th=activation_th,
            deactivation_th=deactivation_th,
            speech_pad_ms=speech_pad_ms,
            min_speech_ms=min_speech_ms,
            max_speech_ms=max_speech_ms,
            partial_frames=partial_frames,
        )
        self.device_name = device
        self.use_onnx = use_onnx and has_onnx
        # Default device annotation for type checkers; will be set in loader
        self.device: torch.device = torch.device("cpu")

        self.speech_buffer: Optional[PcmData] = None

        # Type annotations for inherited attributes from base VAD class
        self.is_speech_active: bool = False
        self._speech_start_time: Optional[float] = None
        self.total_speech_frames: int = 0

        # Verify window size is correct for the Silero model
        if self.sample_rate == 16000 and self.frame_size != 512:
            logger.warning(
                f"Adjusting window_samples from {self.frame_size} to 512, "
                "which is required by Silero VAD at 16kHz"
            )
            self.frame_size = 512
        elif self.sample_rate == 8000 and self.frame_size != 256:
            logger.warning(
                f"Adjusting window_samples from {self.frame_size} to 256, "
                "which is required by Silero VAD at 8kHz"
            )
            self.frame_size = 256

        # Buffered audio at model rate (mono), accumulated across calls
        self._buffered_audio: Optional[PcmData] = None

        # ONNX session and model
        self.onnx_session: Optional["ort.InferenceSession"] = None
        # The Silero VAD torch model is a torch.nn.Module with callable forward
        self.model: Optional[torch.nn.Module] = None
        # ONNX input name if ONNX is used
        self.onnx_input_name: Optional[str] = None

        # Enhanced state tracking for events
        self._current_speech_probability = 0.0
        self._inference_times: list[float] = []  # Track inference performance
        self._speech_start_probability = 0.0
        self._speech_end_probability = 0.0
        self._total_inference_time = 0.0
        self._inference_count = 0
        self._speech_probabilities: list[
            float
        ] = []  # Track probabilities during speech

        # Load the appropriate model
        self._load_model()

    def _load_model(self) -> None:
        """Load the Silero VAD model using torch hub or ONNX runtime."""
        try:
            if self.use_onnx:
                self._load_onnx_model()
            else:
                self._load_torch_model()
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise

    def _load_torch_model(self) -> None:
        """Load the PyTorch version of the Silero VAD model."""
        logger.info("Loading Silero VAD PyTorch model from torch hub")

        # Use torch.hub to load the model and utils
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )

        # Set model to evaluation mode
        assert self.model is not None
        self.model.eval()

        # Try to use the specified device, fall back to CPU if not available
        try:
            self.device = torch.device(self.device_name)
            # Test if CUDA is actually available when requested
            if self.device.type == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = torch.device("cpu")
                self.device_name = "cpu"
            self.model.to(self.device)
            logger.info(f"Using device: {self.device}")
        except Exception as e:
            logger.warning(
                f"Failed to use device {self.device_name}: {e}, falling back to CPU"
            )
            self.device = torch.device("cpu")
            self.device_name = "cpu"
            self.model.to(self.device)

        # Reset states
        self.reset_states()
        logger.info("Silero VAD PyTorch model loaded successfully")

    def _load_onnx_model(self) -> None:
        """Load the ONNX version of the Silero VAD model."""
        if not has_onnx:
            logger.warning("ONNX Runtime not available, falling back to PyTorch model")
            self.use_onnx = False
            self._load_torch_model()
            return

        logger.info("Loading Silero VAD ONNX model")

        try:
            # First load the model with PyTorch to get access to the ONNX export functionality
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )

            # Try to use the specified device for ONNX
            providers = []
            if (
                self.device_name.startswith("cuda")
                and "CUDAExecutionProvider" in ort.get_available_providers()
            ):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                logger.info("Using CUDA for ONNX inference")
            else:
                if self.device_name.startswith("cuda"):
                    logger.warning(
                        "CUDA requested but not available for ONNX, falling back to CPU"
                    )
                providers = ["CPUExecutionProvider"]
                self.device_name = "cpu"

            # Create a session options object and set graph optimization level
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            # Export model to ONNX format in memory and load with ONNX Runtime
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
                # Create dummy input for model
                dummy_input = torch.randn(1, self.window_samples)

                # Export the model
                torch.onnx.export(
                    model,
                    (dummy_input,),
                    tmp.name,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size", 1: "sequence"},
                        "output": {0: "batch_size"},
                    },
                    opset_version=12,
                )

                # Create ONNX session
                self.onnx_session = ort.InferenceSession(
                    tmp.name, sess_options=session_options, providers=providers
                )

                # Get input name
                assert self.onnx_session is not None
                self.onnx_input_name = self.onnx_session.get_inputs()[0].name

            logger.info("Silero VAD ONNX model loaded successfully")

        except Exception as e:
            logger.warning(
                f"Failed to load ONNX model: {e}, falling back to PyTorch model"
            )
            self.use_onnx = False
            self._load_torch_model()

    def reset_states(self) -> None:
        """Reset the model states."""
        # Clear buffered PcmData
        self._buffered_audio = None

    async def is_speech(self, frame: PcmData) -> float:
        """Compute speech probability for a single model window (int16 mono)."""
        try:
            # Expect frame at base sample_rate and frame_size
            x = frame.samples if isinstance(frame.samples, np.ndarray) else np.frombuffer(frame.samples, dtype=np.int16)
            if x.dtype != np.int16:
                x = x.astype(np.int16)
            window = x.astype(np.float32) / 32768.0
            start_time = time.time()
            if self.use_onnx and self.onnx_session is not None:
                onnx_input = window.reshape(1, -1).astype(np.float32)
                ort_inputs = {self.onnx_input_name: onnx_input}
                ort_outputs = self.onnx_session.run(None, ort_inputs)
                speech_prob = float(ort_outputs[0][0])
            else:
                tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    assert self.model is not None
                    speech_prob = float(self.model(tensor, self.sample_rate).item())  # type: ignore[arg-type]

            # Track inference perf
            inf_ms = (time.time() - start_time) * 1000.0
            self._inference_times.append(inf_ms)
            self._total_inference_time += inf_ms
            self._inference_count += 1
            if len(self._inference_times) > 100:
                self._inference_times = self._inference_times[-50:]
            self._current_speech_probability = speech_prob
            return speech_prob
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            return 0.0

    async def _flush_speech_buffer(
        self, user: Optional[Union[Dict[str, Any], "Participant"]] = None
    ) -> None:
        """
        Flush the accumulated speech buffer if it meets minimum length requirements.

        Args:
            user: User metadata to include with emitted audio events
        """
        # Calculate min speech frames based on ms
        min_speech_frames = int(
            self.min_speech_ms * self.sample_rate / 1000 / self.frame_size
        )

        # Serialize buffered PcmData and compute sample count
        speech_bytes = b""
        speech_samples = 0
        if self.speech_buffer is not None:
            speech_bytes = self.speech_buffer.to_bytes()
            speech_samples = (
                len(self.speech_buffer.samples)
                if isinstance(self.speech_buffer.samples, np.ndarray)
                else len(speech_bytes) // 2
            )

        if speech_samples >= min_speech_frames * self.frame_size:
            # Log turn emission at DEBUG level with duration and samples
            duration_ms = (
                self.speech_buffer.duration_ms if self.speech_buffer is not None else 0.0
            )
            logger.debug(
                "Turn emitted",
                extra={"duration_ms": duration_ms, "samples": speech_samples},
            )

            # Calculate average speech probability during this segment
            avg_speech_prob = self._get_avg_speech_probability()

            self.events.send(vad.events.VADAudioEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                audio_data=speech_bytes,
                sample_rate=self.sample_rate,
                audio_format=vad.events.AudioFormat.PCM_S16,
                channels=1,
                duration_ms=duration_ms,
                speech_probability=avg_speech_prob,
                frame_count=speech_samples // self.frame_size,
                user_metadata=user,
            ))

        # Emit speech end event if we were actively detecting speech
        if self.is_speech_active and self._speech_start_time:
            total_speech_duration = (time.time() - self._speech_start_time) * 1000
            self.events.send(
                vad.events.VADSpeechEndEvent(
                    session_id=self.session_id,
                    plugin_name=self.provider_name,
                    speech_probability=self._speech_end_probability,
                    deactivation_threshold=self.deactivation_th,
                    total_speech_duration_ms=total_speech_duration,
                    total_frames=self.total_speech_frames,
                    user_metadata=user,
                )
            )

        # Reset state variables
        self.speech_buffer = None
        self.silence_counter = 0
        self.is_speech_active = False
        self.total_speech_frames = 0
        self.partial_counter = 0
        # Reset enhanced state tracking
        self._speech_start_probability = 0.0
        self._speech_end_probability = 0.0
        self._speech_probabilities = []

    def _get_avg_inference_time(self) -> float:
        """Get average inference time in milliseconds."""
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)

    def _get_avg_speech_probability(self) -> float:
        """Get average speech probability during current segment."""
        if not self._speech_probabilities:
            return self._current_speech_probability
        return sum(self._speech_probabilities) / len(self._speech_probabilities)

    def _get_accumulated_speech_duration(self) -> float:
        """Get accumulated speech duration in milliseconds."""
        if hasattr(self, "_speech_start_time") and self._speech_start_time:
            return (time.time() - self._speech_start_time) * 1000
        return 0.0

    def _get_accumulated_silence_duration(self) -> float:
        """Get accumulated silence duration in milliseconds."""
        return (self.silence_counter * self.frame_size / self.sample_rate) * 1000

    async def _process_frame(
        self, frame: PcmData, user: Optional["Participant"] = None
    ) -> None:
        """
        Process a single audio frame with enhanced Silero-specific event data.
        """
        speech_prob = await self.is_speech(frame)

        # Track speech probabilities during active speech
        if self.is_speech_active:
            self._speech_probabilities.append(speech_prob)
            # Keep only recent probabilities (sliding window)
            if len(self._speech_probabilities) > 100:
                self._speech_probabilities = self._speech_probabilities[-50:]

        # Determine speech state based on asymmetric thresholds
        if self.is_speech_active:
            is_speech = speech_prob >= self.deactivation_th
        else:
            is_speech = speech_prob >= self.activation_th

        # Handle speech start
        if not self.is_speech_active and is_speech:
            self.is_speech_active = True
            self.silence_counter = 0
            self.total_speech_frames = 1
            self.partial_counter = 1
            self._speech_start_time = time.time()
            self._speech_start_probability = speech_prob
            self._speech_probabilities = [speech_prob]  # Reset probability tracking

            self.events.send(
                VADSpeechStartEvent(
                    session_id=self.session_id,
                    plugin_name=self.provider_name,
                    speech_probability=speech_prob,
                    activation_threshold=self.activation_th,
                    frame_count=1,
                    user_metadata=user,
                    audio_data=frame,
                )
            )
            # Initialize speech buffer with first frame
            self.speech_buffer = PcmData(
                samples=frame.samples,
                sample_rate=frame.sample_rate,
                format=frame.format,
            )

        # Handle ongoing speech
        elif self.is_speech_active:
            # Append new frame to accumulated buffer
            if self.speech_buffer is None:
                self.speech_buffer = PcmData(
                    samples=frame.samples,
                    sample_rate=frame.sample_rate,
                    format=frame.format,
                )
            else:
                self.speech_buffer = self.speech_buffer.append(frame)
            self.total_speech_frames += 1
            self.partial_counter += 1

            if self.partial_counter >= self.partial_frames:
                # Serialize current buffer and compute duration
                if self.speech_buffer is not None:
                    current_bytes = self.speech_buffer.to_bytes()
                    current_samples_len = (
                        len(self.speech_buffer.samples)
                        if isinstance(self.speech_buffer.samples, np.ndarray)
                        else len(current_bytes) // 2
                    )
                    current_duration_ms = self.speech_buffer.duration_ms
                else:
                    current_bytes = b""
                    current_samples_len = 0
                    current_duration_ms = 0.0

                self.events.send(
                    vad.events.VADPartialEvent(
                        session_id=self.session_id,
                        plugin_name=self.provider_name,
                        audio_data=current_bytes,
                        sample_rate=self.sample_rate,
                        audio_format=AudioFormat.PCM_S16,
                        channels=1,
                        duration_ms=current_duration_ms,
                        speech_probability=speech_prob,
                        frame_count=current_samples_len // self.frame_size,
                        is_speech_active=True,
                        user_metadata=user,
                    )
                )

                self.partial_counter = 0

            if is_speech:
                # Reset silence counter when speech is detected
                self.silence_counter = 0
            else:
                # Increment silence counter when silence is detected
                self.silence_counter += 1

                # Calculate silence pad frames based on ms
                speech_pad_frames = int(
                    self.speech_pad_ms * self.sample_rate / 1000 / self.frame_size
                )

                # If silence exceeds padding duration, emit audio and reset
                if self.silence_counter >= speech_pad_frames:
                    await self._flush_speech_buffer(user)

            # Calculate max speech frames based on ms
            max_speech_frames = int(
                self.max_speech_ms * self.sample_rate / 1000 / self.frame_size
            )

            # Force flush if speech duration exceeds maximum
            if self.total_speech_frames >= max_speech_frames:
                await self._flush_speech_buffer(user)

    async def reset(self) -> None:
        """Reset the VAD state."""
        await super().reset()
        self.reset_states()
        # Reset enhanced state tracking
        self._current_speech_probability = 0.0
        self._inference_times = []
        self._speech_start_probability = 0.0
        self._speech_end_probability = 0.0
        self._total_inference_time = 0.0
        self._inference_count = 0
        self._speech_probabilities = []

    async def flush(self, user=None) -> None:
        """
        Flush accumulated speech buffer and emit any pending audio events.

        Args:
            user: User metadata to include with emitted audio events
        """
        await super().flush(user)
        # Reset buffer after flushing
        self.reset_states()

    async def close(self) -> None:
        """Release resources used by the model."""
        self.model = None
        if self.onnx_session is not None:
            self.onnx_session = None
        self.reset_states()
        # Reset enhanced state tracking
        self._current_speech_probability = 0.0
        self._inference_times = []
        self._speech_start_probability = 0.0
        self._speech_end_probability = 0.0
        self._total_inference_time = 0.0
        self._inference_count = 0
        self._speech_probabilities = []
