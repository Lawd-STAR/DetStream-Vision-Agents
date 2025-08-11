# stream-agents

Low latency agents. Connect to [Stream's edge network](https://getstream.io/video/) from iOS, Android, Flutter, React Native, React and Unity. 

- Average time to connect:
- P95 latency:

## Features

- Observability & Performance

## Install

```bash
pip install "stream-agents[openai,cartesia]"
```

## Usage Quickstart

```

agent = Agent(
    instructions="Say hi",
    tools=[lookup_stock_price],
    model
    stt
    tts
    turn_detection
)
agent.join(call)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

```

## Observability

You want to track the following metrics

- Number of calls
- https://langfuse.com/

## How do you start an agent/call

- Incoming phone call
- Someone joins a call
- Click a button?
