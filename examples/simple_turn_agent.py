#!/usr/bin/env python3
"""
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from uuid import uuid4

from turn_detection import FalTurnDetection

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from getstream import Stream
from getstream.models import UserRequest
from getstream.plugins.elevenlabs.tts import ElevenLabsTTS
from getstream.plugins.deepgram.stt import DeepgramSTT
from turn_detection import FalTurnDetection

from utils import open_demo

from models import OpenAILLM

# Import the new Agent class from agents2.py
from agents.agents2 import Agent

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")



async def main() -> None:
    """Create a simple agent and join a call."""

    load_dotenv()

    # Initialize Stream client
    client: Stream = Stream.from_env()
    open_ai_key = os.getenv("OPENAI_API_KEY")
    turn_detection = FalTurnDetection(
        buffer_duration=3.0,  # Process 3 seconds of audio at a time
        prediction_threshold=0.7,  # Higher threshold for more confident detections
        mini_pause_duration=0.5,
        max_pause_duration=2.0,
    )


# Create the agent with multiple processors
    agent_user = UserRequest(id=str(uuid4()), name="My happy AI friend")
    agent = Agent(
        llm=OpenAILLM(
            name="gpt-4o",
            instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        ),
        tts=ElevenLabsTTS(),
        stt=DeepgramSTT(),
        turn_detection=turn_detection,
        agent_user=agent_user,
    )

    client.upsert_users(UserRequest(id=agent_user.id, name=agent_user.name))

    try:
        # Join the call - this is the main functionality we're demonstrating
        call = client.video.call("default", str(uuid4()))
        open_demo(client, call.id)
        await agent.join(call)

        # Keep the agent running
        logging.info("🤖 Agent has joined the call. Press Ctrl+C to exit.")

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logging.info("👋 Shutting down agent...")
    except Exception as e:
        logging.error("❌ Error: %s", e)
    finally:
        # Clean up agent resources
        if "agent" in locals():
            try:
                await agent.close()
                logging.info("✅ Agent cleanup completed")
            except Exception as e:
                logging.error(f"❌ Error during cleanup: {e}")



if __name__ == "__main__":
    asyncio.run(main())
