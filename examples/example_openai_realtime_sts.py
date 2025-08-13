#!/usr/bin/env python3
"""
OpenAI Realtime Speech-to-Speech Example for Stream Agents

This example demonstrates how to use the Agent class with OpenAI's real-time
voice API (STS - Speech-to-Speech) for natural conversation in a Stream video call.

Key Features:
- Real-time speech-to-speech communication using OpenAI's Realtime API
- No separate STT/TTS services needed - everything is handled by OpenAI
- Natural conversation flow with interruption support
- Tool calling capabilities
- Configurable voice and instructions

Usage:
    python example_openai_realtime_sts.py

The example will:
1. Create a Stream video call
2. Join as an AI Agent using OpenAI Realtime STS
3. Handle real-time voice conversations
4. Demonstrate tool calling (start closed captions)
5. Provide natural, interruption-capable conversation

Requirements:
    - STREAM_API_KEY and STREAM_SECRET environment variables
    - OPENAI_API_KEY environment variable
    - getstream-python package with openai-realtime support
    - stream-agents package
"""

import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from getstream.stream import Stream
from getstream.models import UserRequest
from utils import open_pronto

# Import Agent and OpenAI STS model
from agents import Agent
from models.openai_sts import OpenAIRealtimeModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Enable detailed logging for OpenAI Realtime
logging.getLogger("getstream_openai.sts").setLevel(logging.INFO)


def create_user(client: Stream, user_id: str, name: str):
    """Create a user in Stream."""
    try:
        user_request = UserRequest(id=user_id, name=name)
        client.upsert_users(user_request)
        logger.info(f"👤 Created user: {name} ({user_id})")
    except Exception as e:
        logger.error(f"❌ Failed to create user {user_id}: {e}")


# Note: Tool calling functionality would be implemented here
# For now, we'll keep this example simple and focus on basic STS functionality


async def main():
    """Main function to run the OpenAI Realtime STS example."""

    print("🎤 Stream Agents - OpenAI Realtime Speech-to-Speech Example")
    print("=" * 65)
    print("This example uses OpenAI's real-time voice API for natural conversation.")
    print("Join the call from your browser and start speaking!")
    print()

    # Load environment variables
    load_dotenv()

    # Check required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY environment variable not set")
        logger.error("Please set your OpenAI API key in your .env file")
        return

    # Initialize Stream client
    try:
        client = Stream.from_env()
        logger.info("✅ Stream client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Stream client: {e}")
        logger.error(
            "Make sure STREAM_API_KEY and STREAM_SECRET are set in your .env file"
        )
        return

    # Create a unique call
    call_id = f"openai-realtime-sts-{str(uuid4())[:8]}"
    call = client.video.call("default", call_id)
    logger.info(f"📞 Call ID: {call_id}")

    # Create user IDs
    participant_user_id = f"participant-{str(uuid4())[:8]}"

    # Create users
    create_user(client, participant_user_id, "Participant")

    # Create tokens
    participant_token = client.create_token(participant_user_id, expiration=3600)

    # Create the call
    try:
        call.get_or_create(data={"created_by_id": participant_user_id})
        logger.info("✅ Call created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create call: {e}")
        return

    # Define tools that the assistant can use (simplified for this example)
    # tools = [
    #     {
    #         "type": "function",
    #         "name": "start_closed_captions",
    #         "description": "Start closed captions for the video call",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {},
    #             "required": [],
    #         },
    #     }
    # ]
    tools = []  # Simplified for now

    # Create OpenAI Realtime STS model
    try:
        sts_model = OpenAIRealtimeModel(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-realtime-preview",
            voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
            instructions=(
                "You are a friendly and helpful AI assistant in a video call. "
                "You MUST speak only in English. Always respond in English language. "
                "Respond naturally and conversationally. Keep your responses concise but engaging. "
                "You can help users with various tasks, including starting closed captions when requested."
            ),
            tools=tools,
            turn_detection={
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 200,
            },
            # Explicitly set input/output audio transcription to English
            input_audio_transcription={"model": "whisper-1", "language": "en"},
        )
        logger.info("✅ OpenAI Realtime STS model created")
    except Exception as e:
        logger.error(f"❌ Failed to create OpenAI Realtime model: {e}")
        return

    # Create Agent with STS model
    agent = Agent(
        instructions=(
            "You are a friendly AI assistant in a video call using OpenAI's real-time voice API. "
            "Engage naturally with participants and help them with their requests."
        ),
        sts_model=sts_model,
        name="OpenAI Realtime Assistant",
    )

    # User creation callback
    def create_agent_user(bot_id: str, name: str):
        create_user(client, bot_id, name)

    # Connection callback to handle tool calls
    async def on_connected(agent_instance, connection):
        """Callback executed after agent connects."""
        logger.info("🔗 Agent connected successfully")
        # For STS mode, the connection handling is managed by the agent itself
        # We don't need to do anything special here

    try:
        # Open browser for participant to join
        open_pronto(client.api_key, participant_token, call_id)

        print()
        print("🎯 OpenAI Realtime STS Agent is ready!")
        print("   • Join the call from your browser")
        print("   • Start speaking naturally - the AI will respond in real-time")
        print("   • Have a natural conversation with the AI")
        print("   • The AI can interrupt and be interrupted naturally")
        print("   • Press Ctrl+C to stop")
        print()

        # Join the call with the Agent
        await agent.join(
            call,
            user_creation_callback=create_agent_user,
            on_connected_callback=on_connected,
        )

    except KeyboardInterrupt:
        logger.info("⏹️  OpenAI Realtime STS agent stopped by user")
    except Exception as e:
        logger.error(f"❌ Error during agent operation: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Cleanup: Delete created users
        try:
            logger.info("🧹 Cleaning up users...")
            client.delete_users([agent.bot_id, participant_user_id])
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

        print()
        print("👋 Thank you for trying the OpenAI Realtime STS example!")
        print("   This demonstrates the power of real-time AI voice conversation")
        print("   in video calls using Stream and OpenAI's latest technology.")


if __name__ == "__main__":
    print("🚀 Starting OpenAI Realtime Speech-to-Speech Agent")
    print("=" * 55)
    print("This example demonstrates:")
    print("• OpenAI Realtime API for speech-to-speech")
    print("• Natural conversation with interruption support")
    print("• Tool calling capabilities")
    print("• Stream video call integration")
    print("• No separate STT/TTS services needed")
    print("=" * 55)

    asyncio.run(main())
