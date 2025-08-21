import asyncio
import logging
import sys
from pathlib import Path
from uuid import uuid4

from edge.edge_transport import StreamEdge

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

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


'''
TODO cleanup:
- stream should be a Transport or similar
- metrics...
- fix call._client = client
'''



async def main() -> None:
    """Create a simple agent and join a call."""

    load_dotenv()



    # Create the agent with multiple processors
    agent_user = UserRequest(id=str(uuid4()), name="My happy AI friend")
    agent = Agent(
        edge=StreamEdge(),
        llm=OpenAILLM(
            name="gpt-4o",
            instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        ),
        tts=ElevenLabsTTS(),
        stt=DeepgramSTT(),
        agent_user=agent_user,
        processors=[],  # Multiple interval processors
    )

    client.upsert_users(UserRequest(id=agent_user.id, name=agent_user.name))

    try:
        # Join the call - this is the main functionality we're demonstrating
        call = client.video.call("default", str(uuid4()))
        # Fix: Set both the client and add stream property for token creation
        call._client = client
        open_demo(client, call.id)
        await agent.join(call)

        # Keep the agent running
        logging.info("🤖 Agent has joined the call. Press Ctrl+C to exit.")
        await agent.finish()
    finally:
        # Fix: agent.close() is async and needs to be awaited
        await agent.close()


if __name__ == "__main__":
    asyncio.run(start_dispatcher(main, log_level="DEBUG"))
