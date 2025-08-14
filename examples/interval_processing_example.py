#!/usr/bin/env python3
"""
Interval Processing Example

This example demonstrates how to use the Agent class with interval processing
for periodic image/video analysis using pre-processors and image processors.
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


# Example pre-processor for demonstration
class MockPreProcessor:
    """Mock pre-processor that simulates video analysis."""
    
    def process(self, frame):
        """Process a video frame and return mock analysis data."""
        if frame is None:
            return {"status": "no_frame", "analysis": "No frame available"}
        
        # Mock analysis - in reality this would do computer vision processing
        return {
            "status": "analyzed",
            "frame_size": getattr(frame, 'size', 'unknown'),
            "timestamp": asyncio.get_event_loop().time(),
            "mock_objects_detected": ["player", "ui_elements", "game_world"],
            "confidence_score": 0.85
        }


# Example image processor
class MockImageProcessor:
    """Mock image processor for additional analysis."""
    
    def process(self, frame):
        """Process image and return additional analysis."""
        return {
            "image_quality": "good",
            "brightness": 0.7,
            "contrast": 0.6,
            "analysis_type": "image_metrics"
        }


async def main() -> None:
    """Create an agent with interval processing capabilities."""
    
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
    
    # Create agent with interval processing
    agent = Agent(
        instructions="You are an AI coach that analyzes video feeds every 5 seconds and provides insights.",
        name="Interval Processing Bot",
        image_interval=5,  # Process every 5 seconds
        pre_processors=[MockPreProcessor()],
        image_processors=[MockImageProcessor()]
    )
    
    logging.info("🤖 Agent created: %s", agent.bot_id)
    logging.info("⏱️ Interval processing configured: every %d seconds", agent.image_interval)
    
    # Create user creation callback for the agent
    def create_bot_user(bot_id: str, bot_name: str):
        create_user(client, bot_id, bot_name)
        logging.info("Created bot user: %s (%s)", bot_id, bot_name)
    
    try:
        # Join the call - interval processing will start automatically
        await agent.join(call, user_creation_callback=create_bot_user)
        
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
