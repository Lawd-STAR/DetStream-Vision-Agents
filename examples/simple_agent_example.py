#!/usr/bin/env python3
"""
Simple Agent Example with Image Capture

This example demonstrates how to create and join a call using the new Agent class
from agents2.py with image capture capabilities. The agent will:

1. Join a Stream video call as an AI assistant
2. Respond to voice input using STT, LLM, and TTS
3. Capture video frames from participants every 3 seconds
4. Save captured frames as JPG images in the 'captured_frames' directory

This shows the core functionality of the Agent class including voice interaction
and video processing capabilities.
"""

import asyncio
import logging
import sys
from pathlib import Path
from uuid import uuid4

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

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


async def main() -> None:
    """Create a simple agent and join a call."""

    load_dotenv()

    # Initialize Stream client
    client: Stream = Stream.from_env()

    # Create multiple interval processors
    image_capture = ImageCapture(output_dir="captured_frames", interval=3)
    audio_logger = AudioLogger(interval=2)  # Log audio every 2 seconds

    # Create the agent with multiple processors
    agent_user = UserRequest(id=str(uuid4()), name="My happy AI friend")
    agent = Agent(
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
        call._client = client
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

        # Display processing summary
        if (
            "image_capture" in locals()
            or "image_analyzer" in locals()
            or "audio_logger" in locals()
        ):
            print()
            print("📊 Processing Summary:")

            if "image_capture" in locals():
                print(f"   • Total frames captured: {image_capture.frame_count}")
                print(f"   • Frames saved to: {image_capture.output_dir.absolute()}")

            if "image_analyzer" in locals():
                print(f"   • Total analyses performed: {image_analyzer.analysis_count}")

            if "audio_logger" in locals():
                print(f"   • Total audio logs: {audio_logger.audio_count}")

            if "image_capture" in locals() and image_capture.frame_count > 0:
                print("   • Captured files:")
                for jpg_file in sorted(image_capture.output_dir.glob("*.jpg")):
                    print(f"     - {jpg_file.name}")


if __name__ == "__main__":
    asyncio.run(main())
