"""
Tests for the Agent class.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from agents import Agent


class MockTool:
    """Mock tool for testing."""
    
    def __init__(self, return_value="mock_result"):
        self.return_value = return_value
    
    def __call__(self, *args, **kwargs):
        return self.return_value


class MockPreProcessor:
    """Mock pre-processor for testing."""
    
    def process(self, data):
        return f"processed_{data}"


class MockModel:
    """Mock AI model for testing."""
    
    async def generate(self, prompt, **kwargs):
        return f"Generated response for: {prompt}"


class MockSTT:
    """Mock Speech-to-Text service."""
    
    async def transcribe(self, audio_data):
        return "transcribed text"


class MockTTS:
    """Mock Text-to-Speech service."""
    
    def __init__(self):
        self.output_track = None
        self.sent_texts = []
    
    def set_output_track(self, track):
        self.output_track = track
    
    async def send(self, text):
        self.sent_texts.append(text)


class MockTurnDetection:
    """Mock turn detection service."""
    
    def detect_turn(self, audio_data):
        return True  # Always return True for testing


class TestAgent:
    """Test cases for the Agent class."""
    
    def test_agent_initialization(self):
        """Test basic agent initialization."""
        instructions = "Test instructions"
        tools = [MockTool()]
        pre_processors = [MockPreProcessor()]
        model = MockModel()
        stt = MockSTT()
        tts = MockTTS()
        turn_detection = MockTurnDetection()
        
        agent = Agent(
            instructions=instructions,
            tools=tools,
            pre_processors=pre_processors,
            model=model,
            stt=stt,
            tts=tts,
            turn_detection=turn_detection,
            name="Test Agent"
        )
        
        assert agent.instructions == instructions
        assert agent.tools == tools
        assert agent.pre_processors == pre_processors
        assert agent.model == model
        assert agent.stt == stt
        assert agent.tts == tts
        assert agent.turn_detection == turn_detection
        assert agent.name == "Test Agent"
        assert agent.bot_id.startswith("agent-")
        assert not agent.is_running()
    
    def test_agent_initialization_with_defaults(self):
        """Test agent initialization with default values."""
        agent = Agent(instructions="Test instructions")
        
        assert agent.instructions == "Test instructions"
        assert agent.tools == []
        assert agent.pre_processors == []
        assert agent.model is None
        assert agent.stt is None
        assert agent.tts is None
        assert agent.turn_detection is None
        assert agent.name == "AI Agent"
        assert agent.bot_id.startswith("agent-")
        assert not agent.is_running()
    
    def test_agent_custom_bot_id(self):
        """Test agent initialization with custom bot ID."""
        custom_id = "custom-bot-123"
        agent = Agent(instructions="Test", bot_id=custom_id)
        
        assert agent.bot_id == custom_id
    
    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test response generation with model."""
        model = MockModel()
        agent = Agent(instructions="Test instructions", model=model)
        
        response = await agent._generate_response("test input")
        
        assert response.startswith("Generated response for:")
        assert "test input" in response
    
    @pytest.mark.asyncio
    async def test_generate_response_without_model(self):
        """Test response generation without model."""
        agent = Agent(instructions="Test instructions")
        
        response = await agent._generate_response("test input")
        
        assert response == ""
    
    @pytest.mark.asyncio
    async def test_generate_response_with_error(self):
        """Test response generation when model raises error."""
        model = Mock()
        model.generate = AsyncMock(side_effect=Exception("Model error"))
        
        agent = Agent(instructions="Test instructions", model=model)
        
        response = await agent._generate_response("test input")
        
        assert "I'm sorry, I encountered an error" in response
    
    @pytest.mark.asyncio
    async def test_generate_greeting(self):
        """Test greeting generation."""
        model = MockModel()
        agent = Agent(instructions="Test instructions", model=model)
        
        greeting = await agent._generate_greeting(2)
        
        assert greeting.startswith("Generated response for:")
        assert "2 participant" in greeting
    
    @pytest.mark.asyncio
    async def test_generate_greeting_without_model(self):
        """Test greeting generation without model."""
        agent = Agent(instructions="Test instructions", name="Test Bot")
        
        greeting = await agent._generate_greeting(1)
        
        assert "Hello everyone! I'm Test Bot" in greeting
    
    @pytest.mark.asyncio
    async def test_generate_participant_greeting(self):
        """Test participant greeting generation."""
        model = MockModel()
        agent = Agent(instructions="Test instructions", model=model)
        
        greeting = await agent._generate_participant_greeting("user-123")
        
        assert greeting.startswith("Generated response for:")
        assert "user-123" in greeting
    
    @pytest.mark.asyncio
    async def test_generate_participant_greeting_without_model(self):
        """Test participant greeting generation without model."""
        agent = Agent(instructions="Test instructions")
        
        greeting = await agent._generate_participant_greeting("user-123")
        
        assert "Welcome user-123!" in greeting
    
    @pytest.mark.asyncio
    async def test_handle_audio_input_without_stt(self):
        """Test audio input handling without STT."""
        agent = Agent(instructions="Test instructions")
        
        # Should not raise an error
        await agent._handle_audio_input(b"fake audio data")
    
    @pytest.mark.asyncio
    async def test_handle_audio_input_with_turn_detection_false(self):
        """Test audio input handling when turn detection returns False."""
        stt = MockSTT()
        turn_detection = Mock()
        turn_detection.detect_turn.return_value = False
        
        agent = Agent(
            instructions="Test instructions",
            stt=stt,
            turn_detection=turn_detection
        )
        
        # Should not process audio when turn detection returns False
        await agent._handle_audio_input(b"fake audio data")
        
        turn_detection.detect_turn.assert_called_once_with(b"fake audio data")
    
    @pytest.mark.asyncio
    async def test_handle_audio_input_full_pipeline(self):
        """Test full audio input processing pipeline."""
        tools = [MockTool("tool_result")]
        pre_processors = [MockPreProcessor()]
        model = MockModel()
        stt = MockSTT()
        tts = MockTTS()
        turn_detection = MockTurnDetection()
        
        agent = Agent(
            instructions="Test instructions",
            tools=tools,
            pre_processors=pre_processors,
            model=model,
            stt=stt,
            tts=tts,
            turn_detection=turn_detection
        )
        
        await agent._handle_audio_input(b"fake audio data")
        
        # Check that TTS received the generated response
        assert len(tts.sent_texts) == 1
        assert tts.sent_texts[0].startswith("Generated response for:")
    
    @pytest.mark.asyncio
    async def test_handle_audio_input_empty_transcription(self):
        """Test audio input handling with empty transcription."""
        stt = Mock()
        stt.transcribe = AsyncMock(return_value="   ")  # Empty/whitespace text
        tts = MockTTS()
        
        agent = Agent(instructions="Test instructions", stt=stt, tts=tts)
        
        await agent._handle_audio_input(b"fake audio data")
        
        # Should not send anything to TTS for empty transcription
        assert len(tts.sent_texts) == 0
    
    @pytest.mark.asyncio
    async def test_handle_new_participant_without_tts(self):
        """Test handling new participant without TTS."""
        agent = Agent(instructions="Test instructions")
        
        # Should not raise an error
        await agent._handle_new_participant("user-123")
    
    @pytest.mark.asyncio
    async def test_handle_new_participant_with_tts(self):
        """Test handling new participant with TTS."""
        tts = MockTTS()
        agent = Agent(instructions="Test instructions", tts=tts)
        
        await agent._handle_new_participant("user-123")
        
        # Should send greeting to TTS
        assert len(tts.sent_texts) == 1
        assert "user-123" in tts.sent_texts[0]
    
    def test_is_running_initially_false(self):
        """Test that agent is not running initially."""
        agent = Agent(instructions="Test instructions")
        assert not agent.is_running()
    
    @pytest.mark.asyncio
    async def test_join_requires_mock_call(self):
        """Test that join method would need mocking for full testing."""
        agent = Agent(instructions="Test instructions")
        
        # We can't easily test the join method without mocking the entire
        # Stream SDK, but we can test that it raises an error when called
        # with invalid arguments
        with pytest.raises(Exception):
            await agent.join(None)  # Invalid call object
    
    def test_pre_processor_pipeline(self):
        """Test pre-processor pipeline."""
        pre_processor1 = MockPreProcessor()
        pre_processor2 = Mock()
        pre_processor2.process = Mock(return_value="final_processed")
        
        agent = Agent(
            instructions="Test instructions",
            pre_processors=[pre_processor1, pre_processor2]
        )
        
        # Simulate the pre-processing pipeline
        data = "original_data"
        for processor in agent.pre_processors:
            data = processor.process(data)
        
        assert data == "final_processed"
        pre_processor2.process.assert_called_once_with("processed_original_data")
    
    def test_tools_integration(self):
        """Test tools integration."""
        tool1 = MockTool("result1")
        tool2 = MockTool("result2")
        
        agent = Agent(instructions="Test instructions", tools=[tool1, tool2])
        
        # Test that tools can be called
        result1 = agent.tools[0]()
        result2 = agent.tools[1]()
        
        assert result1 == "result1"
        assert result2 == "result2"


if __name__ == "__main__":
    pytest.main([__file__])
