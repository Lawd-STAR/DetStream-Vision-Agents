import asyncio
import logging
import sys
from pathlib import Path
from uuid import uuid4


# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from edge.edge_transport import StreamEdge

from dotenv import load_dotenv
from getstream import Stream
from getstream.models import UserRequest
from getstream.plugins.elevenlabs.tts import ElevenLabsTTS
from getstream.plugins.deepgram.stt import DeepgramSTT

from processors.base_processor import ImageCapture, AudioLogger
from utils import open_demo

from models import OpenAILLM

# Import the new Agent class from agents2.py
from agents.agents2 import Agent

# Import the CLI dispatcher
from cli import start_dispatcher



async def main() -> None:
    """Create a simple agent and join a call."""

    load_dotenv()

    # TODO this user creation flow is ugly.
    agent_user = UserRequest(id=str(uuid4()), name="My happy AI friend")
    client = Stream.from_env()
    client.upsert_users(UserRequest(id=agent_user.id, name=agent_user.name))

    # TODO: LLM class 
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
        # turn keeping TODO: enable
        processors=[],  # Multiple interval processors
    )




    try:
        # Join the call - this is the main functionality we're demonstrating
        call = client.video.call("default", str(uuid4()))
        # Open the demo env
        open_demo(call)

        # have the agent join a call/room
        await agent.join(call)
        logging.info("🤖 Agent has joined the call. Press Ctrl+C to exit.")

        # run till the call is ended
        await agent.finish()
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(start_dispatcher(main, log_level="DEBUG"))
