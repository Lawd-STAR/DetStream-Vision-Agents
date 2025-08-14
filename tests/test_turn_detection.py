"""Tests for turn detection module."""

import time
from typing import Dict, Any
from unittest.mock import MagicMock
import pytest
from getstream.models import User
from pyee import EventEmitter

from agents.turn_detection import (
    TurnEvent,
    TurnEventData,
    BaseTurnDetector,
    TurnDetectionProtocol
)



# Example concrete implementation using the base class
class StreamTurnDetector(BaseTurnDetector):
    """Stream.io-specific turn detection implementation."""

    def __init__(self, mini_pause_duration: float, max_pause_duration: float) -> None:
        super().__init__(mini_pause_duration, max_pause_duration)
        self._current_speakers: Dict[str, User] = {}
        self._audio_levels: Dict[str, float] = {}

    def add_participant(self, user: User) -> None:
        """Add a call participant for turn detection tracking."""
        user_id = user.id
        user_name = user.custom.get('name', 'Unknown') if user.custom else 'Unknown'
        self._current_speakers[user_id] = user
        self._audio_levels[user_id] = 0.0
        print(f"Added participant: {user_name} ({user_id})")

    def remove_participant(self, user_id: str) -> None:
        """Remove a participant from turn detection."""
        if user_id in self._current_speakers:
            user = self._current_speakers[user_id]
            user_name = user.custom.get('name', 'Unknown') if user.custom else 'Unknown'
            del self._current_speakers[user_id]
            del self._audio_levels[user_id]
            print(f"Removed participant: {user_name} ({user_id})")

    def start_detection(self) -> None:
        """Start the turn detection process."""
        super().start_detection()
        print(f"Started Stream turn detection with {len(self._current_speakers)} participants")

    def stop_detection(self) -> None:
        """Stop the turn detection process."""
        super().stop_detection()
        print("Stopped Stream turn detection")

    def _process_audio_data(self, audio_data: Any) -> None:
        """Process audio data from Stream call participants."""
        # Your actual audio processing logic would go here
        # For now, this is a placeholder
        pass

    def simulate_speech_event(self, user_id: str, event_type: TurnEvent, **kwargs) -> None:
        """Simulate a speech event for testing purposes."""
        if user_id not in self._current_speakers:
            print(f"User {user_id} not found in participants")
            return

        import time
        user = self._current_speakers[user_id]

        event_data = TurnEventData(
            timestamp=time.time(),
            speaker=user,
            confidence=kwargs.get('confidence', 0.9),
            audio_level=kwargs.get('audio_level', 0.7),
            duration=kwargs.get('duration'),
            custom=kwargs.get('custom')
        )

        self._emit_turn_event(event_type, event_data)


# Example of a completely different implementation that still satisfies the protocol
class SimpleTurnDetector(EventEmitter):
    """Alternative implementation that uses the protocol without inheritance."""

    def __init__(self, mini_pause_duration: float, max_pause_duration: float) -> None:
        super().__init__()
        # Different validation approach
        assert mini_pause_duration > 0, "Mini pause must be positive"
        assert max_pause_duration > mini_pause_duration, "Max pause must be greater than mini pause"

        self._mini_pause = mini_pause_duration
        self._max_pause = max_pause_duration
        self._detecting = False

    @property
    def mini_pause_duration(self) -> float:
        return self._mini_pause

    @property
    def max_pause_duration(self) -> float:
        return self._max_pause

    def is_detecting(self) -> bool:
        return self._detecting

    def start_detection(self) -> None:
        self._detecting = True
        print("Simple detector started")

    def stop_detection(self) -> None:
        self._detecting = False
        print("Simple detector stopped")


def use_any_detector(detector: TurnDetectionProtocol) -> None:
    """Function that works with any object implementing TurnDetectionProtocol."""
    print(f"Using detector with mini_pause: {detector.mini_pause_duration}s")

    # Set up event listener using decorator syntax
    @detector.on('speech_started')
    def on_speech(event_data: TurnEventData) -> None:
        if event_data.speaker and event_data.speaker.custom:
            speaker_name = event_data.speaker.custom.get('name', 'Unknown')
        else:
            speaker_name = "Unknown"
        print(f"Detected speech from {speaker_name}")

    detector.start_detection()
    print(f"Detection active: {detector.is_detecting()}")




class TestBaseTurnDetector:
    """Tests for BaseTurnDetector class."""

    def test_init_valid_durations(self):
        """Test initialization with valid durations."""
        detector = BaseTurnDetector(0.5, 2.0)
        assert detector.mini_pause_duration == 0.5
        assert detector.max_pause_duration == 2.0
        assert not detector.is_detecting()

    def test_init_invalid_mini_pause(self):
        """Test initialization with invalid mini pause duration."""
        with pytest.raises(ValueError, match="mini_pause_duration must be positive"):
            BaseTurnDetector(-0.5, 2.0)
        
        with pytest.raises(ValueError, match="mini_pause_duration must be positive"):
            BaseTurnDetector(0, 2.0)

    def test_init_invalid_max_pause(self):
        """Test initialization with invalid max pause duration."""
        with pytest.raises(ValueError, match="max_pause_duration must be positive"):
            BaseTurnDetector(0.5, -2.0)
        
        with pytest.raises(ValueError, match="max_pause_duration must be positive"):
            BaseTurnDetector(0.5, 0)

    def test_init_mini_greater_than_max(self):
        """Test initialization with mini pause greater than max pause."""
        with pytest.raises(ValueError, match="mini_pause_duration must be less than max_pause_duration"):
            BaseTurnDetector(2.0, 0.5)
        
        with pytest.raises(ValueError, match="mini_pause_duration must be less than max_pause_duration"):
            BaseTurnDetector(2.0, 2.0)

    def test_start_stop_detection(self):
        """Test starting and stopping detection."""
        detector = BaseTurnDetector(0.5, 2.0)
        assert not detector.is_detecting()
        
        detector.start_detection()
        assert detector.is_detecting()
        
        detector.stop_detection()
        assert not detector.is_detecting()

    def test_emit_turn_event(self):
        """Test emitting turn events."""
        detector = BaseTurnDetector(0.5, 2.0)
        event_received = []
        
        @detector.on(TurnEvent.SPEECH_STARTED.value)
        def on_speech(sppech):
            event_received.append(sppech)
        
        event_data = TurnEventData(
            timestamp=time.time(),
            confidence=0.9
        )
        detector._emit_turn_event(TurnEvent.SPEECH_STARTED, event_data)
        
        assert len(event_received) == 1
        assert event_received[0] == event_data


class TestStreamTurnDetector:
    """Tests for StreamTurnDetector class."""

    def test_initialization(self):
        """Test StreamTurnDetector initialization."""
        detector = StreamTurnDetector(0.5, 2.0)
        assert detector.mini_pause_duration == 0.5
        assert detector.max_pause_duration == 2.0
        assert not detector.is_detecting()
        assert len(detector._current_speakers) == 0
        assert len(detector._audio_levels) == 0

    def test_add_participant(self, capsys):
        """Test adding a participant."""
        detector = StreamTurnDetector(0.5, 2.0)
        user = User(id="test-user", role="participant", banned=False, online=True, custom={'name': 'John Doe'}, teams_role={})
        
        detector.add_participant(user)
        
        assert "test-user" in detector._current_speakers
        assert detector._current_speakers["test-user"] == user
        assert "test-user" in detector._audio_levels
        assert detector._audio_levels["test-user"] == 0.0
        
        captured = capsys.readouterr()
        assert "Added participant: John Doe (test-user)" in captured.out

    def test_remove_participant(self, capsys):
        """Test removing a participant."""
        detector = StreamTurnDetector(0.5, 2.0)
        user = User(id="test-user", role="participant", banned=False, online=True, custom={'name': 'John Doe'}, teams_role={})
        
        detector.add_participant(user)
        detector.remove_participant("test-user")
        
        assert "test-user" not in detector._current_speakers
        assert "test-user" not in detector._audio_levels
        
        captured = capsys.readouterr()
        assert "Removed participant: John Doe (test-user)" in captured.out

    def test_remove_nonexistent_participant(self):
        """Test removing a participant that doesn't exist."""
        detector = StreamTurnDetector(0.5, 2.0)
        # Should not raise an error
        detector.remove_participant("nonexistent-user")

    def test_start_stop_detection(self, capsys):
        """Test starting and stopping detection with participants."""
        detector = StreamTurnDetector(0.5, 2.0)
        user1 = User(id="user1", role="participant", banned=False, online=True, custom={'name': 'User One'}, teams_role={})
        user2 = User(id="user2", role="participant", banned=False, online=True, custom={'name': 'User Two'}, teams_role={})
        
        detector.add_participant(user1)
        detector.add_participant(user2)
        
        detector.start_detection()
        assert detector.is_detecting()
        
        captured = capsys.readouterr()
        assert "Started Stream turn detection with 2 participants" in captured.out
        
        detector.stop_detection()
        assert not detector.is_detecting()
        
        captured = capsys.readouterr()
        assert "Stopped Stream turn detection" in captured.out

    def test_simulate_speech_event(self, capsys):
        """Test simulating speech events."""
        detector = StreamTurnDetector(0.5, 2.0)
        user = User(id="test-user", role="participant", banned=False, online=True, custom={'name': 'John Doe'}, teams_role={})
        detector.add_participant(user)
        
        events_received = []
        
        @detector.on(TurnEvent.SPEECH_STARTED.value)
        def on_speech(event_data):
            events_received.append(event_data)
        
        detector.simulate_speech_event(
            "test-user",
            TurnEvent.SPEECH_STARTED,
            confidence=0.95,
            audio_level=0.8,
            duration=1.5,
            custom={"extra": "data"}
        )
        
        assert len(events_received) == 1
        event = events_received[0]
        assert event.speaker == user
        assert event.confidence == 0.95
        assert event.audio_level == 0.8
        assert event.duration == 1.5
        assert event.custom == {"extra": "data"}

    def test_simulate_speech_event_nonexistent_user(self, capsys):
        """Test simulating speech event for nonexistent user."""
        detector = StreamTurnDetector(0.5, 2.0)
        
        detector.simulate_speech_event("nonexistent", TurnEvent.SPEECH_STARTED)
        
        captured = capsys.readouterr()
        assert "User nonexistent not found in participants" in captured.out


class TestSimpleTurnDetector:
    """Tests for SimpleTurnDetector class."""

    def test_initialization(self):
        """Test SimpleTurnDetector initialization."""
        detector = SimpleTurnDetector(0.3, 1.5)
        assert detector.mini_pause_duration == 0.3
        assert detector.max_pause_duration == 1.5
        assert not detector.is_detecting()

    def test_init_invalid_durations(self):
        """Test initialization with invalid durations."""
        with pytest.raises(AssertionError, match="Mini pause must be positive"):
            SimpleTurnDetector(-0.3, 1.5)
        
        with pytest.raises(AssertionError, match="Mini pause must be positive"):
            SimpleTurnDetector(0, 1.5)
        
        with pytest.raises(AssertionError, match="Max pause must be greater than mini pause"):
            SimpleTurnDetector(1.5, 0.3)
        
        with pytest.raises(AssertionError, match="Max pause must be greater than mini pause"):
            SimpleTurnDetector(1.5, 1.5)

    def test_start_stop_detection(self, capsys):
        """Test starting and stopping detection."""
        detector = SimpleTurnDetector(0.3, 1.5)
        
        detector.start_detection()
        assert detector.is_detecting()
        captured = capsys.readouterr()
        assert "Simple detector started" in captured.out
        
        detector.stop_detection()
        assert not detector.is_detecting()
        captured = capsys.readouterr()
        assert "Simple detector stopped" in captured.out


class TestProtocolCompatibility:
    """Tests for protocol compatibility between different implementations."""

    def test_use_any_detector_with_stream_detector(self, capsys):
        """Test that StreamTurnDetector works with use_any_detector function."""
        stream_detector = StreamTurnDetector(0.5, 2.0)
        user = User(id="test-user", role="participant", banned=False, online=True, custom={'name': 'John Doe'}, teams_role={})
        stream_detector.add_participant(user)
        
        # Capture output to verify function works
        use_any_detector(stream_detector)
        
        captured = capsys.readouterr()
        assert "Using detector with mini_pause: 0.5s" in captured.out
        assert "Started Stream turn detection with 1 participants" in captured.out
        assert "Detection active: True" in captured.out
        
        # Simulate an event to test the listener
        stream_detector.simulate_speech_event("test-user", TurnEvent.SPEECH_STARTED)
        captured = capsys.readouterr()
        assert "Detected speech from John Doe" in captured.out

    def test_use_any_detector_with_simple_detector(self, capsys):
        """Test that SimpleTurnDetector works with use_any_detector function."""
        simple_detector = SimpleTurnDetector(0.3, 1.5)
        
        use_any_detector(simple_detector)
        
        captured = capsys.readouterr()
        assert "Using detector with mini_pause: 0.3s" in captured.out
        assert "Simple detector started" in captured.out
        assert "Detection active: True" in captured.out

    def test_both_satisfy_protocol(self):
        """Test that both implementations satisfy the TurnDetectionProtocol."""
        stream_detector = StreamTurnDetector(0.5, 2.0)
        simple_detector = SimpleTurnDetector(0.3, 1.5)
        
        # Both should have required properties
        for detector in [stream_detector, simple_detector]:
            assert hasattr(detector, 'mini_pause_duration')
            assert hasattr(detector, 'max_pause_duration')
            assert hasattr(detector, 'is_detecting')
            assert hasattr(detector, 'start_detection')
            assert hasattr(detector, 'stop_detection')
            assert hasattr(detector, 'on')
            assert hasattr(detector, 'emit')
            
            # All methods should be callable
            assert callable(detector.is_detecting)
            assert callable(detector.start_detection)
            assert callable(detector.stop_detection)
            assert callable(detector.on)
            assert callable(detector.emit)


class TestTurnEventData:
    """Tests for TurnEventData dataclass."""

    def test_minimal_initialization(self):
        """Test TurnEventData with minimal required fields."""
        event_data = TurnEventData(timestamp=1234567890.0)
        assert event_data.timestamp == 1234567890.0
        assert event_data.speaker is None
        assert event_data.duration is None
        assert event_data.confidence is None
        assert event_data.audio_level is None
        assert event_data.custom is None

    def test_full_initialization(self):
        """Test TurnEventData with all fields."""
        user = User(id="test", role="participant", banned=False, online=True, custom={'name': 'Test User'}, teams_role={})
        event_data = TurnEventData(
            timestamp=1234567890.0,
            speaker=user,
            duration=2.5,
            confidence=0.95,
            audio_level=0.7,
            custom={"key": "value"}
        )
        assert event_data.timestamp == 1234567890.0
        assert event_data.speaker == user
        assert event_data.duration == 2.5
        assert event_data.confidence == 0.95
        assert event_data.audio_level == 0.7
        assert event_data.custom == {"key": "value"}


class TestTurnEvent:
    """Tests for TurnEvent enum."""

    def test_enum_values(self):
        """Test that TurnEvent enum has expected values."""
        assert TurnEvent.SPEECH_STARTED.value == "speech_started"
        assert TurnEvent.SPEECH_ENDED.value == "speech_ended"
        assert TurnEvent.TURN_STARTED.value == "turn_started"
        assert TurnEvent.TURN_ENDED.value == "turn_ended"
        assert TurnEvent.MINI_PAUSE_DETECTED.value == "mini_pause_detected"
        assert TurnEvent.MAX_PAUSE_REACHED.value == "max_pause_reached"