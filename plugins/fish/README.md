# Fish Audio Text-to-Speech Plugin

A high-quality Text-to-Speech (TTS) plugin for Vision Agents that uses the Fish Audio API.

## Installation

```bash
pip install vision-agents-plugins-fish
```

## Usage

```python
from vision_agents.plugins.fish import TTS
from getstream.video.rtc.audio_track import AudioStreamTrack

# Initialize with API key from environment variable
tts = TTS()

# Or specify API key directly
tts = TTS(api_key="your_fish_audio_api_key")

# Create an audio track to output speech
track = AudioStreamTrack(framerate=16000)
tts.set_output_track(track)

# Register event handlers
@tts.on("audio")
def on_audio(audio_data, user):
    print(f"Received audio chunk: {len(audio_data)} bytes")

# Send text to be converted to speech
await tts.send("Hello, this is a test of the Fish Audio text-to-speech plugin.")
```

## Configuration Options

- `api_key`: Fish Audio API key (default: reads from FISH_AUDIO_API_KEY environment variable)
- `reference_id`: Optional reference voice ID to use for synthesis
- `base_url`: Optional custom API endpoint (default: uses Fish Audio's default endpoint)

## Reference Audio

Fish Audio supports using reference audio for voice cloning:

```python
from vision_agents.plugins.fish import TTS

# Using a reference voice ID
tts = TTS(reference_id="your_reference_voice_id")

# Or pass reference audio dynamically when sending text
# (See Fish Audio SDK documentation for advanced usage)
```

## Requirements

- Python 3.10+
- fish-audio-sdk>=2025.4.2

