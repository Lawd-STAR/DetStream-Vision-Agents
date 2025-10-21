# AWS Plugin for Vision Agents

AWS (Bedrock) LLM integration for Vision Agents framework with support for both standard and realtime interactions.

## Installation

```bash
pip install vision-agents-plugins-aws
```

## Usage

### Standard LLM Usage

This example shows how to use qwen3 on bedrock for the LLM.

```python
agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Friendly AI"),
    instructions="Be nice to the user",
    llm=aws.LLM(model="qwen.qwen3-32b-v1:0"),
    tts=cartesia.TTS(),
    stt=deepgram.STT(),
    turn_detection=smart_turn.TurnDetection(buffer_duration=2.0, confidence_threshold=0.5),
)
```

The full example is available in example/aws_qwen_example.py

### Realtime Text/Image Usage

Nova sonic audio realtime STS is also supported:

```python    
agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Story Teller AI"),
    instructions="Tell a story suitable for a 7 year old about a dragon and a princess",
    llm=aws.Realtime(),
)
```

