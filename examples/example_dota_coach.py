#!/usr/bin/env python3
"""
Dota 2 Coach Example for Stream Agents

This example demonstrates how to use the Agent class with Gemini STS for
real-time Dota 2 coaching with computer vision analysis and game API integration.

Key Features:
- Gemini Live API for natural speech-to-speech coaching
- YOLO-based computer vision analysis (similar to workout assistant)
- Real-time game statistics from Valve's official Dota 2 Web API
- Interval-based processing (configurable intervals)
- Multimodal AI coaching with vision + data + voice
- Advanced game state analysis (team fights, farming opportunities, danger assessment)
- Comprehensive player performance metrics from official match data

Usage:
    python example_dota_coach.py [--game-id GAME_ID] [--interval SECONDS]

The example will:
1. Create a Stream video call
2. Join as an AI Coach using Gemini STS
3. Analyze gameplay every second using computer vision
4. Fetch live game statistics from Dota API
5. Provide real-time voice coaching based on @ai-dota-coaching.md guidelines
6. React to team fights, farming opportunities, and performance issues

Requirements:
    - STREAM_API_KEY and STREAM_SECRET environment variables
    - GOOGLE_API_KEY or GEMINI_API_KEY environment variable
    - STEAM_API_KEY environment variable (optional, will use mock data if not provided)
    - getstream-python package
    - stream-agents package with processors
    - ultralytics package (for YOLO models)
    - aiohttp package (for Valve API calls)
"""

import argparse
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

# Import Agent and components
from agents import Agent
from models.gemini_sts import GeminiSTS
from processors.yolo_processor import YOLOProcessor
from processors.dota_api import dota_api

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Enable detailed logging for components
logging.getLogger("GeminiSTS").setLevel(logging.INFO)
logging.getLogger("YOLOProcessor").setLevel(logging.INFO)
logging.getLogger("DotaAPI").setLevel(logging.INFO)


def create_user(client: Stream, user_id: str, name: str):
    """Create a user in Stream."""
    try:
        user_request = UserRequest(id=user_id, name=name)
        client.upsert_users(user_request)
        logger.info(f"👤 Created user: {name} ({user_id})")
    except Exception as e:
        logger.error(f"❌ Failed to create user {user_id}: {e}")


async def main(game_id: str = "match_123456", interval: int = 1):
    """Main function to run the Dota 2 coach example."""

    print("🎮 Stream Agents - Dota 2 AI Coach Example")
    print("=" * 50)
    print("This example uses Gemini Live API for real-time Dota 2 coaching.")
    print("The AI coach will analyze your gameplay and provide live feedback!")
    print()

    # Load environment variables
    load_dotenv()

    # Check required environment variables
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        logger.error("❌ GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
        logger.error("Please set your Google AI API key in your .env file")
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
    call_id = f"dota-coach-{str(uuid4())[:8]}"
    call = client.video.call("default", call_id)
    logger.info(f"📞 Call ID: {call_id}")

    # Create user IDs
    player_user_id = f"player-{str(uuid4())[:8]}"

    # Create users
    create_user(client, player_user_id, "Dota Player")

    # Create tokens
    player_token = client.create_token(player_user_id, expiration=3600)

    # Create the call
    try:
        call.get_or_create(data={"created_by_id": player_user_id})
        logger.info("✅ Call created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create call: {e}")
        return

    # Read coaching instructions
    coaching_instructions = ""
    try:
        with open("ai-dota-coaching.md", "r") as f:
            coaching_instructions = f.read()
        logger.info("📖 Loaded coaching instructions")
    except FileNotFoundError:
        logger.warning("⚠️ ai-dota-coaching.md not found, using basic instructions")
        coaching_instructions = """
        You are an expert Dota 2 coach. Provide real-time feedback on gameplay.
        Be encouraging but point out mistakes. Focus on positioning, farming, and team fighting.
        """

    # Create Gemini STS model
    try:
        sts_model = GeminiSTS(
            api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            model="gemini-2.0-flash-exp",
            instructions=(
                f"You are an expert Dota 2 coach providing real-time gameplay analysis. "
                f"Follow these coaching guidelines:\n\n{coaching_instructions}\n\n"
                f"You receive multimodal data every {interval} seconds including:\n"
                f"- Live video frames from the game\n"
                f"- Computer vision analysis (heroes, items, positioning)\n"
                f"- Live game statistics (KDA, farm, items)\n\n"
                f"Provide specific, actionable coaching advice based on what you see and the data. "
                f"Be encouraging but honest about mistakes. Keep responses concise and focused."
            ),
            voice_config={
                "voice": "alloy",  # Can be customized based on Gemini's available voices
                "speed": 1.1,  # Slightly faster for real-time coaching
            },
        )
        logger.info("✅ Gemini STS model created")
    except Exception as e:
        logger.error(f"❌ Failed to create Gemini STS model: {e}")
        return

    # Create pre-processors
    try:
        # YOLO-based computer vision processor
        yolo_processor = YOLOProcessor(
            model_path="yolo11n.pt",  # Will use general YOLO model, can be replaced with Dota-specific model
            confidence=0.6,
            device="cpu",  # Change to "cuda" if GPU available
            classes=None,  # None = detect all classes, can specify specific classes for Dota objects
        )

        # Valve Dota 2 API processor
        dota_api_processor = dota_api(
            game_id=game_id,
            api_key=os.getenv(
                "STEAM_API_KEY"
            ),  # Optional: will use mock data if not provided
            player_id=None,  # Optional: Steam ID of specific player to focus on
        )

        logger.info("✅ Pre-processors created")
    except Exception as e:
        logger.error(f"❌ Failed to create pre-processors: {e}")
        return

    # Create Agent with the exact syntax requested
    agent = Agent(
        instructions="Roast my in-game performance in a funny but encouraging manner. Follow coaching tips in @ai-dota-coaching.md",
        pre_processors=[yolo_processor, dota_api_processor],
        image_interval=interval,  # Process every interval seconds
        sts_model=sts_model,
        name="Dota 2 AI Coach",
        # turn_detection=your_turn_detector  # Commented out as requested
    )

    # User creation callback
    def create_agent_user(bot_id: str, name: str):
        create_user(client, bot_id, name)

    # Connection callback for additional setup
    async def on_connected(agent_instance, connection):
        """Callback executed after agent connects."""
        logger.info("🔗 Dota coach connected and ready to analyze gameplay!")

        # Send initial coaching message
        if hasattr(agent_instance.sts_model, "send_message"):
            await agent_instance.sts_model.send_message(
                f"Welcome to your personal Dota 2 coaching session! I'm analyzing your gameplay every {interval} seconds. "
                f"Let's see what you've got - time to climb those ranks! Start playing and I'll give you real-time feedback."
            )

    try:
        # Open browser for player to join
        open_pronto(client.api_key, player_token, call_id)

        print()
        print("🎯 Dota 2 AI Coach is ready!")
        print("   • Join the call from your browser")
        print("   • Start your Dota 2 match")
        print("   • Share your screen to show the game")
        print(f"   • AI coach will analyze gameplay every {interval} seconds")
        print("   • Get real-time coaching advice via voice")
        print("   • Press Ctrl+C to stop")
        print()
        print("🎮 Coaching Features:")
        print(
            "   • YOLO-based computer vision analysis (heroes, creeps, towers, items)"
        )
        print("   • Valve Dota 2 Web API integration for official match statistics")
        print("   • Advanced game state detection (team fights, farming opportunities)")
        print(
            "   • Comprehensive performance analysis (KDA, GPM, XPM, damage, objectives)"
        )
        print("   • Real-time multimodal feedback combining vision + game data + voice")
        print(
            "   • Encouraging but honest performance roasting with specific recommendations"
        )
        print()

        # Join the call with the Agent
        await agent.join(
            call,
            user_creation_callback=create_agent_user,
            on_connected_callback=on_connected,
        )

    except KeyboardInterrupt:
        logger.info("⏹️  Dota 2 AI coach stopped by user")
    except Exception as e:
        logger.error(f"❌ Error during coaching session: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Cleanup: Delete created users
        try:
            logger.info("🧹 Cleaning up users...")
            client.delete_users([agent.bot_id, player_user_id])
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

        print()
        print("🏆 Coaching Session Complete!")
        print("   Thanks for using the Dota 2 AI Coach!")
        print("   Keep practicing and you'll dominate the Ancient!")


if __name__ == "__main__":
    print("🚀 Starting Dota 2 AI Coach")
    print("=" * 35)
    print("This example demonstrates:")
    print("• Gemini Live API for speech-to-speech coaching")
    print("• Real-time computer vision analysis")
    print("• Live game statistics integration")
    print("• Interval-based multimodal processing")
    print("• Expert Dota 2 coaching with humor")
    print("=" * 35)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Dota 2 AI Coach Example for Stream Agents"
    )
    parser.add_argument(
        "--game-id",
        default="match_123456",
        help="Dota 2 match ID to track (default: match_123456)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Analysis interval in seconds (default: 1)",
    )

    args = parser.parse_args()

    # Run the example with parsed arguments
    asyncio.run(main(game_id=args.game_id, interval=args.interval))
