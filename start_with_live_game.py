#!/usr/bin/env python3
"""
Start Dota 2 Coach with Live Game

This script helps you start the Dota 2 coach with either:
1. A live game from Valve's API (requires Steam API key)
2. A recent/example game ID for testing
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def fetch_live_games_with_api(api_key: str) -> List[Dict]:
    """Fetch live games using Steam API."""
    try:
        from processors.dota_api import DotaAPI

        temp_api = DotaAPI("temp", api_key)
        live_games = await temp_api._fetch_live_games()
        return live_games or []
    except Exception as e:
        logger.error(f"Error fetching live games: {e}")
        return []


def get_example_matches() -> List[Dict]:
    """Get some example recent match IDs for testing."""
    return [
        {
            "match_id": "7853972481",
            "description": "Recent Professional Match (Example)",
            "league_name": "Example Tournament",
            "teams": "Team A vs Team B",
        },
        {
            "match_id": "7853972482",
            "description": "Recent Ranked Match (Example)",
            "league_name": "Ranked Matchmaking",
            "teams": "Radiant vs Dire",
        },
        {
            "match_id": "7853972483",
            "description": "High MMR Match (Example)",
            "league_name": "Immortal Ranked",
            "teams": "High Skill Game",
        },
    ]


async def main():
    """Main function to start the coach with a live game."""
    print("🎮 Dota 2 Coach - Live Game Starter")
    print("=" * 45)

    # Check for Steam API key
    api_key = os.getenv("STEAM_API_KEY")

    if api_key:
        print("✅ Steam API key found! Fetching live games...")
        try:
            live_games = await fetch_live_games_with_api(api_key)

            if live_games:
                print(f"🎯 Found {len(live_games)} live games!")
                print()

                # Display live games
                for i, game in enumerate(live_games[:5], 1):  # Show max 5 games
                    match_id = game.get("match_id", "Unknown")
                    league_name = game.get("league_name", "Unknown League")
                    spectators = game.get("spectators", 0)
                    game_time = game.get("game_time", 0)

                    minutes = game_time // 60
                    seconds = game_time % 60

                    print(f"{i}. Match {match_id}")
                    print(f"   League: {league_name}")
                    print(f"   Time: {minutes}:{seconds:02d}")
                    print(f"   Spectators: {spectators:,}")
                    print()

                # Ask user to select
                while True:
                    try:
                        choice = input(
                            f"Select a live game (1-{len(live_games[:5])}) or 'e' for examples: "
                        ).strip()
                        if choice.lower() == "e":
                            break

                        game_index = int(choice) - 1
                        if 0 <= game_index < len(live_games[:5]):
                            selected_match_id = live_games[game_index].get("match_id")
                            print(f"🚀 Starting with live match: {selected_match_id}")
                            await start_coach(selected_match_id)
                            return
                        else:
                            print(f"❌ Please enter 1-{len(live_games[:5])} or 'e'")
                    except ValueError:
                        print("❌ Please enter a valid number or 'e'")
            else:
                print("😔 No live games found. Using example matches instead...")

        except Exception as e:
            print(f"⚠️ Error fetching live games: {e}")
            print("Using example matches instead...")
    else:
        print("ℹ️  No Steam API key found.")
        print("   To fetch real live games, get a Steam API key from:")
        print("   https://steamcommunity.com/dev/apikey")
        print("   Then set: export STEAM_API_KEY='your_key_here'")
        print()
        print("📝 Using example match IDs for now...")

    # Show example matches
    print()
    print("🎯 Example Matches:")
    example_matches = get_example_matches()

    for i, match in enumerate(example_matches, 1):
        print(f"{i}. Match {match['match_id']}")
        print(f"   {match['description']}")
        print(f"   League: {match['league_name']}")
        print(f"   Teams: {match['teams']}")
        print()

    # Ask user to select an example match
    while True:
        try:
            choice = input(
                f"Select an example match (1-{len(example_matches)}) or 'q' to quit: "
            ).strip()
            if choice.lower() == "q":
                print("👋 Goodbye!")
                return

            match_index = int(choice) - 1
            if 0 <= match_index < len(example_matches):
                selected_match = example_matches[match_index]
                match_id = selected_match["match_id"]
                print(f"🚀 Starting with example match: {match_id}")
                await start_coach(match_id)
                return
            else:
                print(f"❌ Please enter 1-{len(example_matches)} or 'q'")
        except ValueError:
            print("❌ Please enter a valid number or 'q'")


async def start_coach(match_id: str):
    """Start the Dota 2 coach with the specified match ID."""
    print()
    print(f"🎮 Starting Dota 2 AI Coach with Match ID: {match_id}")
    print("=" * 60)
    print("🔥 Real-time coaching with YOLO vision + Valve API + Gemini STS!")
    print()

    try:
        # Import and run the coach
        from examples.example_dota_coach import main as coach_main

        await coach_main(game_id=match_id, interval=3)  # 3-second intervals

    except KeyboardInterrupt:
        print("\n⏹️  Coach stopped by user")
    except Exception as e:
        logger.error(f"❌ Error running coach: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
