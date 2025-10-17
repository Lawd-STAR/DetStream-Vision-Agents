# AWS Bedrock Plugin for Vision Agents

AWS Bedrock LLM integration for Vision Agents framework with support for both standard and realtime interactions.

## Installation

```bash
pip install vision-agents-plugins-bedrock
```

## Usage

### Standard LLM Usage

```python
from vision_agents.plugins import bedrock

# Initialize the Bedrock LLM
llm = bedrock.LLM(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-east-1"
)

# Simple response
response = await llm.simple_response("Hello, how are you?")
print(response.text)
```

### Realtime Audio/Video Usage

```python
from vision_agents.plugins import bedrock

# Initialize Bedrock Realtime with Nova Sonic for speech-to-speech
realtime = bedrock.Realtime(
    model="us.amazon.nova-sonic-v1:0",
    region_name="us-east-1",
    sample_rate=16000
)

# Connect to the session
await realtime.connect()

# Send text message
await realtime.simple_response("Describe what you see")

# Send audio
pcm_data = PcmData(...)  # Your audio data
await realtime.simple_audio_response(pcm_data)

# Watch video track
await realtime._watch_video_track(video_track)

# Close when done
await realtime.close()
```

## Configuration

The plugin uses boto3 for AWS authentication. You can configure credentials using:
- AWS credentials file (~/.aws/credentials)
- Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- IAM roles (when running on AWS services)

## Supported Models

### Standard Models (LLM class)
All AWS Bedrock models are supported, including:
- Claude 3.5 models (anthropic.claude-*)
- Amazon Titan models (amazon.titan-*)
- Meta Llama models (meta.llama-*)
- And more

### Realtime Models (Realtime class)
Realtime audio/video models optimized for speech-to-speech:
- **Amazon Nova Sonic (us.amazon.nova-sonic-v1:0)** - Primary model for realtime interactions with ultra-low latency
- Amazon Nova Lite (us.amazon.nova-lite-v1:0)
- Amazon Nova Micro (us.amazon.nova-micro-v1:0)
- Amazon Nova Pro (us.amazon.nova-pro-v1:0)
- And other Nova models

**Note:** Nova Sonic is specifically designed for realtime speech-to-speech conversations and is the recommended default for the Realtime class.

See [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/) for available models.

