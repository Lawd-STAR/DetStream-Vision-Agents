# Vogent Turn Keeping Plugin

A multimodal turn detection plugin for Vision Agents using the [Vogent Turn model](https://github.com/vogent/vogent-turn).

## Overview

Vogent Turn is a fast and accurate turn detection system that combines audio intonation and text context to determine when a speaker has completed their turn in a conversation. This plugin integrates the Vogent Turn model into the Vision Agents framework.

The model uses:
- **Audio Encoder**: Whisper-Tiny for processing audio features
- **Text Model**: SmolLM-135M for understanding conversation context
- **Multimodal Fusion**: Combines both modalities for accurate predictions

For more information, see the [Vogent Turn GitHub repository](https://github.com/vogent/vogent-turn).

## Installation

```bash
pip install vision-agents-plugins-vogent
```

## Usage

```python
from vision_agents.plugins.vogent import TurnDetection

# Initialize with optional configuration
turn_detector = TurnDetection(
    model_name="vogent/Vogent-Turn-80M",  # HuggingFace model ID
    buffer_duration=2.0,                  # Seconds of audio to buffer
    confidence_threshold=0.5,             # Threshold for turn completion
    compile_model=True                    # Use torch.compile for speed
)

# Register event handlers
@turn_detector.on("turn_started")
def on_turn_started(event_data):
    print(f"Turn started: {event_data.speaker_id}")

@turn_detector.on("turn_ended")
def on_turn_ended(event_data):
    print(f"Turn ended: {event_data.speaker_id} (confidence: {event_data.confidence:.3f})")

# Start detection
turn_detector.start()

# Process audio (typically called in a streaming loop)
await turn_detector.process_audio(pcm_data, user_id="user123")

# Stop detection
turn_detector.stop()
```

## Configuration Options

- `model_name`: HuggingFace model ID (default: "vogent/Vogent-Turn-80M")
- `buffer_duration`: Duration in seconds to buffer audio before processing (default: 2.0)
- `confidence_threshold`: Probability threshold for turn completion (default: 0.5)
- `sample_rate`: Audio sample rate in Hz (default: 16000)
- `channels`: Number of audio channels (default: 1)
- `compile_model`: Use torch.compile for faster inference (default: True)

## Current Limitations

**Text Context**: The Vogent Turn model is designed to be multimodal, using both audio and text context (`prev_line` and `curr_line`). Currently, this plugin uses empty strings for text context as a placeholder. Future versions will integrate with STT (Speech-to-Text) to provide real-time transcription context for improved accuracy.

## Audio Requirements

- **Sample rate**: 16kHz (automatically resampled from 48kHz)
- **Channels**: Mono (automatically converted if stereo)
- **Format**: int16 PCM
- **Duration**: Processes in 2-second chunks by default

