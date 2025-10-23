"""
Basic tests for Vogent turn detection.
"""

import pytest
import numpy as np

from vision_agents.plugins.vogent import TurnDetection


def test_turn_detection_init():
    """Test basic initialization."""
    detector = TurnDetection()
    assert detector is not None
    assert not detector.is_detecting()
    assert detector._confidence_threshold == 0.5


def test_turn_detection_init_with_params():
    """Test initialization with custom parameters."""
    detector = TurnDetection(
        model_name="vogent/Vogent-Turn-80M",
        buffer_duration=3.0,
        confidence_threshold=0.7,
        compile_model=False,
    )
    assert detector is not None
    assert detector.buffer_duration == 3.0
    assert detector._confidence_threshold == 0.7


def test_turn_detection_start_stop():
    """Test start/stop functionality."""
    detector = TurnDetection()
    
    # Initially not detecting
    assert not detector.is_detecting()
    
    # Start detection
    detector.start()
    assert detector.is_detecting()
    
    # Stop detection
    detector.stop()
    assert not detector.is_detecting()
    
    # Calling start/stop multiple times should be safe
    detector.start()
    detector.start()
    assert detector.is_detecting()
    
    detector.stop()
    detector.stop()
    assert not detector.is_detecting()


@pytest.mark.asyncio
async def test_process_audio_when_not_detecting():
    """Test that process_audio does nothing when not detecting."""
    detector = TurnDetection()
    
    # Should not raise an error, just return early
    await detector.process_audio(None, "user123")


def test_infer_channels():
    """Test channel inference from format strings."""
    detector = TurnDetection()
    
    assert detector._infer_channels("mono") == 1
    assert detector._infer_channels("s16") == 1
    assert detector._infer_channels("int16") == 1
    assert detector._infer_channels("pcm_s16le") == 1
    assert detector._infer_channels("stereo") == 2
    assert detector._infer_channels("unknown") == 1  # defaults to mono

