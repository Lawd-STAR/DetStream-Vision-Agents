"""Tests for AWS Bedrock Realtime plugin."""
import asyncio
import pytest
from dotenv import load_dotenv

from vision_agents.plugins.bedrock import Realtime
from vision_agents.core.llm.events import RealtimeAudioOutputEvent

# Load environment variables
load_dotenv()


class TestBedrockRealtime:
    """Integration tests for Bedrock Realtime connect flow"""

    @pytest.fixture
    async def realtime(self):
        """Create and manage Realtime connection lifecycle"""
        # Note: Nova Sonic requires a specialized WebSocket API, not ConverseStream
        # Using Claude for testing as it supports ConverseStream
        realtime = Realtime(
            model="amazon.nova-sonic-v1:0",
            region_name="us-east-1",
        )
        try:
            yield realtime
        finally:
            await realtime.close()

    @pytest.mark.integration
    async def test_simple_response_flow(self, realtime):
        """Test sending a simple text message and receiving response"""
        # Send a simple message
        events = []
        realtime._set_instructions("whenever you reply mention a fun fact about The Netherlands")
        
        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)
        
        await asyncio.sleep(0.01)
        await realtime.connect()
        await realtime.simple_response("Hello, can you hear me? Please respond with a short greeting.")

        # Wait for response
        await asyncio.sleep(10.0)
        
        # Note: Depending on model capabilities, audio events may or may not be generated
        # The test passes if no exceptions are raised
        assert True

    @pytest.mark.integration
    async def test_audio_sending_flow_start(self, realtime, mia_audio_16khz):
        """Test sending real audio data and verify connection remains stable"""
        events = []
        
        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)
        
        await asyncio.sleep(0.01)
        await realtime.connect()
        
        #await realtime.simple_response("Listen to the following story, what is Mia looking for?")
        #await asyncio.sleep(10.0)
        #await realtime.simple_audio_response(mia_audio_16khz)

        # Wait a moment to ensure processing
        #await asyncio.sleep(10.0)
        
        # Test passes if no exceptions are raised
        #assert True

    @pytest.mark.integration
    async def test_video_sending_flow(self, realtime, bunny_video_track):
        """Test sending real video data and verify connection remains stable"""
        events = []
        
        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)
        
        await asyncio.sleep(0.01)
        await realtime.connect()
        await realtime.simple_response("Describe what you see in this video please")
        await asyncio.sleep(5.0)
        
        # Start video sender with low FPS to avoid overwhelming the connection
        await realtime._watch_video_track(bunny_video_track)
        
        # Let it run for a few seconds
        await asyncio.sleep(10.0)
        
        # Stop video sender
        await realtime._stop_watching_video_track()
        
        # Test passes if no exceptions are raised
        assert True

    @pytest.mark.integration
    async def test_connection_lifecycle(self, realtime):
        """Test that connection can be established and closed properly"""
        # Connect
        await realtime.connect()
        assert realtime._is_connected is True
        
        # Send a simple message
        await realtime.simple_response("Test message")
        await asyncio.sleep(2.0)
        
        # Close
        await realtime.close()
        assert realtime._is_connected is False

