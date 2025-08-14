#!/usr/bin/env python3
"""
Simple Agent Example

This example demonstrates the simplest way to create and join a call using the new Agent class
from agents2.py. This is a minimal example that shows just the core functionality.
"""

import asyncio
import logging
import os
import sys
import webbrowser
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlencode

from getstream import Stream
from getstream.plugins.elevenlabs.tts import ElevenLabsTTS
from getstream.video.rtc.utils import open_browser
from stt import DeepgramSTT

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from getstream.models import UserRequest
from getstream.stream import Stream

from utils import open_demo

from models import OpenAILLM

# Import the new Agent class from agents2.py
from agents.agents2 import Agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")



async def main() -> None:
    """Create a simple agent and join a call."""
    
    load_dotenv()
    
    # Initialize Stream client
    client: Stream = Stream.from_env()

    # Create the simplest possible agent
    agent = Agent(
        llm=OpenAILLM(name="gpt-5-2025-08-07"),
        tts=ElevenLabsTTS(),
        stt=DeepgramSTT(),
    )

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
        pass


if __name__ == "__main__":
    asyncio.run(main())
