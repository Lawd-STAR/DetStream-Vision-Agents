# Kickboxing Coach AI Agent Example

This example demonstrates an advanced AI coach for kickboxing training using the Agent class with real-time pose detection and Gemini Live for interactive coaching.

## Features

- 🥊 **Real-time Pose Detection**: Uses YOLO pose detection to analyze kickboxing form and technique
- 🎤 **Voice Interaction**: Speech-to-speech coaching via Gemini Live
- 📊 **Advanced Analysis**: Focuses on stance, guard, hip rotation, and movement patterns
- 🎯 **Personalized Feedback**: AI coach provides specific, actionable coaching feedback
- 📸 **Visual Annotations**: Highlights key body points and skeletal structure for analysis

## Requirements

### Environment Variables

Create a `.env` file in the project root with:

```bash
# Stream API credentials
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret

# Google Gemini API key for AI coaching
GOOGLE_API_KEY=your_google_api_key
```

### Dependencies

The required dependencies are defined in `pyproject.toml`. Key packages include:

- `ultralytics` - YOLO pose detection
- `opencv-python` - Computer vision processing
- `google-genai` - Gemini Live integration
- `getstream[webrtc]` - Video calling infrastructure

### Model Files

The example requires the YOLO pose detection model:
- `yolo11n-pose.pt` - YOLO pose detection model (included in examples directory)

## Usage

1. **Install dependencies**:
   ```bash
   cd stream-agents
   pip install -e .
   ```

2. **Set up environment variables** in `.env` file

3. **Run the example**:
   ```bash
   cd examples
   python kickboxing_agent_example.py
   ```

4. **Join the call** from your browser using the URL displayed in the console

5. **Start training** - the AI coach will:
   - Analyze your kickboxing form in real-time
   - Provide voice feedback on technique
   - Focus on key areas like stance, guard, and movement

## How It Works

### Architecture

The example uses the clean Agent class architecture from `agents2.py` with:

1. **KickboxingPoseProcessor**: An `IntervalProcessor` that:
   - Runs YOLO pose detection every second
   - Annotates video frames with skeletal overlay
   - Highlights critical points (wrists, ankles, hips)
   - Queues frames for AI analysis

2. **GeminiLiveLLM**: A custom LLM implementation that:
   - Connects to Gemini Live for speech-to-speech interaction
   - Receives pose-annotated video frames
   - Provides expert kickboxing coaching feedback
   - Uses a specialized coaching prompt

3. **Background Tasks**:
   - Frame processing and annotation
   - Audio response handling
   - Visual data streaming to AI

### Key Components

- **Pose Detection**: Uses YOLO11n-pose model for real-time skeletal tracking
- **Coaching AI**: Gemini Live analyzes visual data and provides voice feedback
- **Video Processing**: Processes frames at 1 FPS for optimal performance
- **Audio Pipeline**: Bidirectional audio for natural conversation

## Coaching Focus Areas

The AI coach analyzes and provides feedback on:

- **Stance and Footwork**: Base position and foot placement
- **Guard Position**: Hand and arm positioning for defense
- **Hip Rotation**: Core engagement and power generation
- **Kick Mechanics**: Chamber, extension, and recovery
- **Punch Technique**: Shoulder alignment and form
- **Transitions**: Flow between offensive and defensive positions
- **Balance and Fluidity**: Overall movement quality

## Customization

You can customize the coaching behavior by:

1. **Adjusting Processing Interval**: Change the `interval` parameter in `KickboxingPoseProcessor`
2. **Modifying Coaching Prompt**: Update the `system_instruction` in `GeminiLiveLLM`
3. **Adding New Processors**: Extend with additional analysis capabilities
4. **Changing Voice**: Modify the `voice_name` in Gemini configuration

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -e .`
2. **Model Not Found**: Verify `yolo11n-pose.pt` is in the examples directory
3. **API Key Issues**: Check that environment variables are set correctly
4. **Performance Issues**: Adjust processing interval or image size for better performance

### Performance Tips

- The example is optimized for CPU inference
- Frame processing runs at 1 FPS to balance quality and performance
- Queue management prevents memory buildup during processing

## Example Output

When running, you'll see logs like:

```
🥊 Kickboxing pose processor initialized with yolo11n-pose.pt
🤖 Connected to Gemini Live for kickboxing coaching
🥊 Kickboxing Coach AI is now active!
🥊 Processed pose frame #1 for user-12345
🎤 Trainee: How's my stance looking?
🥊 Coach: Your stance is too narrow! Spread your feet shoulder-width apart...
```

This example demonstrates the power of combining real-time computer vision with advanced AI for personalized coaching applications.
