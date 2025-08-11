# stream-agents

Low latency agents. Connect to [Stream's edge network](https://getstream.io/video/) from iOS, Android, Flutter, React Native, React and Unity. 

- Average time to connect:
- P95 latency:

Want to go even faster? Use the Go version of this library.

## Features

- Observability & Performance

## Install

```bash
pip install "stream-agents[openai,cartesia]"
```

## Video Quickstart

```python

agent = Agent(
    instructions="Roast my in-game performance in a funny but encouraging manner",
    tools=[dota_api("gameid")],
    pre_processors=[Roboflow()]
    model
    stt
    tts
    turn_detection
)
agent.join(call)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

```

## Demo

* Vercel
* Chat & Images

## Usage Quickstart

```python

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

## Low Latency Recommendation

* 

## Observability

You want to track the following metrics

- Number of calls
- https://langfuse.com/

## How do you start an agent/call

- Incoming phone call
- Someone joins a call
- Click a button?


## roadmap

- Agent
- Observability
- TTS interface
- SST
- Turn keeping
- Model

- https://openrouter.ai/


