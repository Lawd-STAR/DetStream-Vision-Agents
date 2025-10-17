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

### Realtime Text/Image Usage

```python
from vision_agents.plugins import bedrock

# Initialize Bedrock Realtime (uses ConverseStream API)
realtime = bedrock.Realtime(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1"
)

# Connect to the session
await realtime.connect()

# Send text message
await realtime.simple_response("Describe what you see")

# Send audio (PCM format)
pcm_data = PcmData(...)  # Your PCM audio data
await realtime.simple_audio_response(pcm_data)

# Watch video track (for image frames)
await realtime._watch_video_track(video_track)

# Close when done
await realtime.close()
```

**Note on Audio**: Audio input is now supported following the [Nova Sonic pattern](https://github.com/aws-samples/amazon-nova-samples/blob/main/speech-to-speech/sample-codes/console-python/nova_sonic.py#L296). Audio is sent as PCM format in the conversation messages. Note that audio support depends on the model being used - Nova Sonic specifically requires a specialized WebSocket API (not ConverseStream) for full speech-to-speech capabilities.

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
The Realtime class uses AWS Bedrock's ConverseStream API and supports models that work with this API:
- **Claude 3 models (anthropic.claude-3-*)** - Recommended for text and image streaming
- Other text/image models that support ConverseStream

**Note on Nova Sonic**: Amazon Nova Sonic (us.amazon.nova-sonic-v1:0) is designed for speech-to-speech conversations but requires a specialized WebSocket API, not ConverseStream. The current Realtime implementation focuses on text/image streaming via ConverseStream. For Nova Sonic integration, see [AWS Nova Sonic examples](https://github.com/aws-samples/amazon-nova-samples/blob/main/speech-to-speech/).

See [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/) for available models.

