# Open Agents by Stream

Low latency video (and voice) AI agents on [Stream's edge network](https://getstream.io/video/).

* Average time to join: 500ms
* 95% percentile audio transport latency: 30ms

Open Agent library. Goal is to support most of our video/audio competitors. (See adding support section)
SDKs for React, Android, iOS, Flutter, React, React Native and Unity.


## 🚀 Features

- ✅ **Low Latency**: Built for real-time video interactions
- ✅ **AI-Powered**: OpenAI GPT integration with extensible model system
- ✅ **Voice & Vision**: Support for speech-to-text, text-to-speech, and computer vision
- ✅ **Extensible**: Plugin architecture for tools, pre-processors, and AI services
- ✅ **Production Ready**: Comprehensive testing, error handling, and observability

## 📦 Installation

```bash
# Install dependencies using uv
uv add openai python-dotenv getstreamt

# Or with pip
pip install openai python-dotenv getstream
```

## ⚡ Quick Start - Cluely style AI

```python
from agents import Agent

# Roboflow for object detection (finetuned)
# load docs from @ai-dota-coaching.md
# use speech to speech (STS) gemini model
agent = Agent(
    pre_processors=[Roboflow(), dota_api("gameid")],
    interval=1 second,
    llm=GeminiSTS(), 
    # turn_detection=your_turn_detector
)

# Join a Stream video call
await agent.join(call)

# gemini is available at
agent.llm.client

# history at
agent.conversation
```




## ⚡ Quick Start - Video AI Coach

```python
from agents import Agent

# Roboflow for object detection (finetuned)
# load docs from @ai-dota-coaching.md
# use speech to speech (STS) gemini model
agent = Agent(
    pre_processors=[Roboflow(), dota_api("gameid")],
    interval=1 second,
    llm=GeminiSTS(), 
    # turn_detection=your_turn_detector
)

# Join a Stream video call
await agent.join(call)
```

## Video Transform

Roboflow server side transform of video


```python
from agents import Agent


# Create an agent with the exact syntax you requested
agent = Agent(
    video_transformer=RoboflowTransform() # transform live video from python (instead of on-device). use for AI avatars
)

# Join a Stream video call
await agent.join(call)
```

## Quick Example - Voice AI (TTS, STT) and STS

```python
from agents import Agent
from models import OpenAILLM

# Create an AI model
llm = OpenAILLM(
    api_key="<your-api-key>",
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello there"},
    ],
    default_temperature=0.8,
)

# Create an agent with the exact syntax you requested
agent = Agent(
    stt=your_stt_service,
    pre_processors=[Roboflow()],
    llm=llm,
    tts=your_tts_service,
    turn_detection=your_turn_detector
)

sts_model = OpenAIRealtime(
    api_key="<your-api-key>",
    model="gpt-4o-realtime-preview",
    voice="Pluck"
)
speechToSpeechAgent = Agent(
    sts=sts_model,
)

# Join a Stream video call
await agent.join(call)
```


## 📚 Examples

Other example to build
- Simple image
- Simple video capture
- Simple audio capture
- Moderation example (AWS image. comarketing)
- SIP with twilio
- Cluely clone

Check out the [`examples/`](./examples/) directory for complete working examples:

- **`examples/main.py`** - Basic TTS bot with ElevenLabs
- **`examples/example_agent.py`** - Agent framework with tools and pre-processors  
- **`examples/example_openai_model.py`** - OpenAI model integration
- **`examples/example_agent_with_openai.py`** - Complete AI agent with OpenAI

```bash
# Run any example
python examples/example_agent_with_openai.py
```

## 🔧 Configuration

Create a `.env` file with your API keys:

```bash
# Stream API (required)
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret
EXAMPLE_BASE_URL=https://pronto-staging.getstream.io

# ElevenLabs TTS (for voice examples)
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# OpenAI (for AI model examples)
OPENAI_API_KEY=your_openai_api_key
```

## 🏗️ Architecture

### Core Components

- **`agents/`** - Agent framework and protocols
- **`models/`** - AI model implementations (OpenAI, extensible to others)
- **`examples/`** - Working examples and demos
- **`tests/`** - Comprehensive test suite

### Agent System

```python
agent = Agent(
    tools=[external_api_tool],           # External API integrations
    pre_processors=[data_processor],     # Input data processing  
    llm=ai_model,                      # AI model for responses
    stt=speech_to_text,                  # Speech recognition
    tts=text_to_speech,                  # Voice synthesis
    turn_detection=turn_detector         # Conversation management
)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/test_agent.py -v
pytest tests/test_models.py -v
```

## 🛣️ Roadmap

- ✅ Agent framework with protocol-based architecture
- ✅ OpenAI model integration
- ✅ Tools and pre-processors system
- ✅ Stream video call integration
- ✅ Comprehensive testing
- 🔄 Speech-to-text integration
- 🔄 Turn detection system
- 🔄 Memory and context management
- 🔄 Additional model providers (Anthropic, Cohere, etc.)
- 🔄 Advanced observability and metrics
- 🔄 Production deployment guides

## 🤝 Dev Guidelines

### Light wrapping

AI is changing daily. This makes it important to use light wrapping. IE

```python
tts = ElevenLabsTTS(client=ElevenLabsClient())
```

Note how the ElevenLabsTTS handles standardization.
But if the init for ElevenLabsClient changes, nothing breaks.
If features are added to the client, you can use them easily via tts.client

### Typing

Avoid using Union types or complicated composite types.
Keep typing simple. Use the PcmAudio type instead of bytes when passing around audio.
This prevents mistakes related to different audio formats. 

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.


## Observability

- Traces and metrics go to Prometheus and OpenTelemetry
- Metrics on performance of TTS, STT, LLM, Turn detection and connection to realtime edge.
- Integration with external LLM observability solutions