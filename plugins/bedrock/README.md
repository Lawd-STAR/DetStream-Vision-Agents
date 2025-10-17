# AWS Bedrock Plugin for Vision Agents

AWS Bedrock LLM integration for Vision Agents framework.

## Installation

```bash
pip install vision-agents-plugins-bedrock
```

## Usage

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

## Configuration

The plugin uses boto3 for AWS authentication. You can configure credentials using:
- AWS credentials file (~/.aws/credentials)
- Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- IAM roles (when running on AWS services)

## Supported Models

All AWS Bedrock models are supported, including:
- Claude 3.5 models (anthropic.claude-*)
- Amazon Titan models (amazon.titan-*)
- Meta Llama models (meta.llama-*)
- And more

See [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/) for available models.

