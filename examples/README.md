# Stream Agents Examples

This directory contains example implementations demonstrating how to use the Stream Agents framework.

## 📁 Examples Overview

### 🎯 Basic Examples

#### `main.py`
**Original TTS Bot Example**
- Simple text-to-speech bot using ElevenLabs
- Demonstrates basic Stream video call integration
- Shows how to create users, join calls, and send audio

```bash
python examples/main.py
```

#### `example_agent.py`
**Agent Framework Example**
- Demonstrates the new Agent class with tools and pre-processors
- Shows the requested Agent syntax with mock implementations
- Includes example Dota API tool and Roboflow pre-processor

```bash
python examples/example_agent.py
```

### 🤖 AI Model Integration

#### `example_openai_model.py`
**OpenAI Model Standalone Example**
- Tests the OpenAI model implementation independently
- Demonstrates text generation, chat, and streaming
- Shows how to integrate OpenAI models with agents

```bash
python examples/example_openai_model.py
```

#### `example_agent_with_openai.py`
**Complete AI Agent with OpenAI**
- Full integration of Agent + OpenAI model + Stream calls
- Demonstrates the complete requested syntax:
  ```python
  model = OpenAIModel(name="gpt-4o-mini", default_temperature=0.8)
  agent = Agent(
      instructions="Roast my in-game performance...",
      tools=[dota_api("gameid")],
      pre_processors=[Roboflow()],
      model=model,
      tts=tts
  )
  await agent.join(call)
  ```

```bash
python examples/example_agent_with_openai.py
```

## 🚀 Prerequisites

### Environment Variables
Create a `.env` file in the project root with:

```bash
# Stream API (required for all examples)
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret
EXAMPLE_BASE_URL=https://pronto-staging.getstream.io

# ElevenLabs TTS (required for voice examples)
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# OpenAI (required for AI model examples)
OPENAI_API_KEY=your_openai_api_key
```

### Dependencies
All dependencies are managed via `uv`. The examples use:
- `getstream` - Stream video API client
- `elevenlabs` - Text-to-speech service
- `openai` - OpenAI API client
- `python-dotenv` - Environment variable loading

## 📋 Running Examples

1. **Set up environment variables** (see above)

2. **Run any example:**
   ```bash
   # From the project root
   python examples/main.py
   python examples/example_agent.py
   python examples/example_openai_model.py
   python examples/example_agent_with_openai.py
   ```

3. **Join the video call** in your browser when prompted

4. **Interact with the agent** (depending on the example)

## 🛠️ Example Features

### Agent Capabilities
- ✅ **Instructions**: Custom AI personality and behavior
- ✅ **Tools**: External API integrations (Dota API example)
- ✅ **Pre-processors**: Input data processing (Roboflow example)
- ✅ **AI Models**: OpenAI GPT integration
- ✅ **TTS**: ElevenLabs text-to-speech
- ✅ **STT**: Speech-to-text (interface ready)
- ✅ **Turn Detection**: Conversation management (interface ready)

### Stream Integration
- ✅ **Video Calls**: Create and join Stream video calls
- ✅ **User Management**: Automatic user creation and cleanup
- ✅ **Audio Streaming**: Real-time audio transmission
- ✅ **Event Handling**: Participant join/leave events
- ✅ **Browser Integration**: Automatic browser opening

## 🎯 Next Steps

1. **Customize the examples** for your use case
2. **Add your own tools** and pre-processors
3. **Integrate different AI models** using the Model protocol
4. **Add STT and turn detection** services
5. **Deploy to production** with proper error handling

## 🐛 Troubleshooting

### Common Issues

**"API key not found"**
- Ensure all required environment variables are set in `.env`

**"WebSocket connection closed"**
- This is normal when stopping the agent (Ctrl+C)
- The cleanup warning can be ignored

**"Browser didn't open"**
- Manually open the printed URL in your browser
- Check that `EXAMPLE_BASE_URL` is set correctly

**"OpenAI API error"**
- Verify your OpenAI API key is valid
- Check you have sufficient credits
- Ensure the model name is correct (e.g., "gpt-4o-mini")

### Getting Help

- Check the main project README
- Review the test files for usage patterns
- Open an issue for bugs or feature requests
