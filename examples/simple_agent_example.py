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

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from getstream.models import UserRequest
from getstream.stream import Stream

# Import the new Agent class from agents2.py
from agents.agents2 import Agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def create_user(client: Stream, user_id: str, name: str) -> None:
    """Create a user with Stream."""
    user_request = UserRequest(id=user_id, name=name)
    client.upsert_users(user_request)


def open_browser(api_key: str, token: str, call_id: str) -> str:
    """Helper function to open browser with Stream call link."""
    base_url = f"{os.getenv('EXAMPLE_BASE_URL', 'https://getstream.io/video/demos')}/join/"
    params = {"api_key": api_key, "token": token, "skip_lobby": "true"}

    url = f"{base_url}{call_id}?{urlencode(params)}"
    print(f"🌐 Opening browser to: {url}")

    try:
        webbrowser.open(url)
        print("✅ Browser opened successfully!")
    except Exception as e:
        print(f"❌ Failed to open browser: {e}")
        print(f"Please manually open this URL: {url}")

    return url


async def main() -> None:
    """Create a simple agent and join a call."""
    
    load_dotenv()
    
    # Initialize Stream client
    client = Stream.from_env()
    
    # Create a human user for testing
    human_id = f"user-{uuid4()}"
    create_user(client, human_id, "Human User")
    
    # Create user token for browser access
    token = client.create_token(human_id, expiration=3600)
    
    # Create the call
    call_id = str(uuid4())
    call = client.video.call("default", call_id)
    call.get_or_create(data={"created_by_id": human_id})
    
    logging.info("📞 Call created: %s", call_id)
    
    # Open browser automatically
    open_browser(client.api_key, token, call_id)
    
    # Create the simplest possible agent
    agent = Agent(
        instructions="You are a helpful AI assistant. Be friendly and conversational.",
        name="Simple Bot"
    )
    
    logging.info("🤖 Agent created: %s", agent.bot_id)
    
    # Create user creation callback for the agent
    def create_bot_user(bot_id: str, bot_name: str):
        create_user(client, bot_id, bot_name)
        logging.info("Created bot user: %s (%s)", bot_id, bot_name)
    
    try:
        # Join the call - this is the main functionality we're demonstrating
        await agent.join(call, user_creation_callback=create_bot_user)
        
        # Keep the agent running
        logging.info("🤖 Agent has joined the call. Press Ctrl+C to exit.")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("👋 Shutting down agent...")
    except Exception as e:
        logging.error("❌ Error: %s", e)
    finally:
        # Cleanup users
        try:
            client.delete_users([human_id, agent.bot_id])
            logging.info("🧹 Cleanup completed")
        except Exception as e:
            logging.error("❌ Cleanup error: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
