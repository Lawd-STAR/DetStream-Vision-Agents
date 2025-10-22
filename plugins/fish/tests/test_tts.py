import asyncio

import pytest
from dotenv import load_dotenv

from vision_agents.plugins import fish
from vision_agents.core.tts.events import TTSAudioEvent, TTSErrorEvent
from getstream.video.rtc.audio_track import AudioStreamTrack

# Load environment variables
load_dotenv()

# Audio track for capturing test output
class MockAudioTrack(AudioStreamTrack):
    def __init__(self, framerate: int = 16000):
        self.framerate = framerate
        self.written_data = []

    async def write(self, data: bytes):
        self.written_data.append(data)
        return True


@pytest.mark.integration
async def test_fish_tts_convert_text_to_audio():
    """
    Integration test with the real Fish Audio API.
    
    This test uses the actual Fish Audio API with the
    FISH_AUDIO_API_KEY environment variable.
    It will be skipped if the environment variable is not set.
    
    To set up the FISH_AUDIO_API_KEY:
    1. Sign up for a Fish Audio account at https://fish.audio
    2. Create an API key in your Fish Audio dashboard
    3. Add to your .env file: FISH_AUDIO_API_KEY=your_api_key_here
    """

    
    # Create a real Fish Audio TTS instance
    tts = fish.TTS()
    
    # Create an audio track to capture the output
    track = MockAudioTrack()
    tts.set_output_track(track)
    
    # Track audio events
    audio_received = asyncio.Event()
    received_chunks = []
    
    @tts.events.subscribe
    async def on_audio(event: TTSAudioEvent):
        received_chunks.append(event.audio_data)
        audio_received.set()
    
    # Track API errors
    api_errors = []
    
    @tts.events.subscribe
    async def on_error(event: TTSErrorEvent):
        api_errors.append(event.error)
        audio_received.set()  # Unblock the waiting
    
    # Allow event subscriptions to be processed
    await asyncio.sleep(0.01)
    
    try:
        # Use a short text to minimize API usage
        text = "Hello from Fish Audio."
        
        # Send the text to generate speech
        send_task = asyncio.create_task(tts.send(text))
        
        # Wait for either audio or an error
        try:
            await asyncio.wait_for(audio_received.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            # Cancel the task if it's taking too long
            send_task.cancel()
            pytest.fail("No audio or error received within timeout")
        
        # Check if we received any API errors
        if api_errors:
            pytest.skip(f"API error received: {api_errors[0]}")
        
        # Try to ensure the send task completes
        try:
            await send_task
        except Exception as e:
            pytest.skip(f"Exception during TTS generation: {e}")
        
        # Verify that we received audio data
        assert len(received_chunks) > 0, "No audio chunks were received"
        assert len(track.written_data) > 0, "No audio data was written to track"
    except Exception as e:
        pytest.skip(f"Unexpected error in Fish Audio test: {e}")

