import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv
from getstream.plugins.deepgram.stt import DeepgramSTT
from getstream.plugins.elevenlabs.tts import ElevenLabsTTS
from stream_agents.turn_detection import FalTurnDetection
from stream_agents.llm import OpenAILLM

from stream_agents import Agent, Stream, StreamEdge, start_dispatcher, open_demo


async def main() -> None:
    """Create a simple agent and join a call."""

    load_dotenv()
    client = Stream.from_env()

    # create the AI user
    agent_user = client.create_user(name="My happy AI friend")

    # Create the agent
    agent = Agent(
        edge=StreamEdge(), # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=agent_user, # the user name etc for the agent
        # tts, llm, stt more. see the realtime example for sts
        llm=OpenAILLM(
            name="gpt-4o",
            instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        ),
        tts=ElevenLabsTTS(),
        stt=DeepgramSTT(),
        # turn keeping
        turn_detection=FalTurnDetection(),
        # processors can fetch extra data, check images/audio data or transform video
        processors=[],
    )

    try:
        # Join the call - this is the main functionality we're demonstrating
        call = client.video.call("default", str(uuid4()))
        # Open the demo env
        open_demo(call)

        # have the agent join a call/room
        await agent.join(call)

        # run till the call is ended
        await agent.finish()
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(start_dispatcher(main))
