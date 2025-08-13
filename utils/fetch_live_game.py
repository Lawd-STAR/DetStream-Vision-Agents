#!/usr/bin/env python3
"""
Live Game Fetcher for Dota 2 Coach

This utility fetches current live Dota 2 games from Valve's API
and can start the coach with a specific game ID.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.dota_api import DotaAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def fetch_live_games(api_key: Optional[str] = None) -> List[Dict]:
    """Fetch current live Dota 2 games."""
    # Create a temporary DotaAPI instance to use its live games method
    temp_api = DotaAPI("temp", api_key)

    if temp_api.use_mock:
        logger.warning("⚠️ No Steam API key available, cannot fetch live games")
        return []

    live_games = await temp_api._fetch_live_games()
    return live_games or []


def format_game_info(game: Dict) -> str:
    """Format game information for display."""
    match_id = game.get("match_id", "Unknown")
    league_name = game.get("league_name", "Unknown League")
    spectators = game.get("spectators", 0)
    game_time = game.get("game_time", 0)

    # Format game time
    minutes = game_time // 60
    seconds = game_time % 60
    time_str = f"{minutes}:{seconds:02d}"

    # Get team info
    radiant_team = game.get("radiant_team", {})
    dire_team = game.get("dire_team", {})
    radiant_name = radiant_team.get("team_name", "Radiant")
    dire_name = dire_team.get("team_name", "Dire")

    return (
        f"Match {match_id}: {radiant_name} vs {dire_name}\n"
        f"  League: {league_name}\n"
        f"  Game Time: {time_str}\n"
        f"  Spectators: {spectators:,}"
    )


async def main():
    """Main function to fetch and display live games."""
    print("🎮 Fetching Live Dota 2 Games")
    print("=" * 40)

    # Check for Steam API key
    api_key = os.getenv("STEAM_API_KEY")
    if not api_key:
        print("❌ No STEAM_API_KEY environment variable found!")
        print("   Please set your Steam API key to fetch live games.")
        print("   You can get one from: https://steamcommunity.com/dev/apikey")
        print()
        print("   Example: export STEAM_API_KEY='your_api_key_here'")
        return

    try:
        print("🔍 Fetching live games from Valve API...")
        live_games = await fetch_live_games(api_key)

        if not live_games:
            print("😔 No live games found at the moment.")
            print("   Live games are typically professional/league matches.")
            print("   Try again later when tournaments are running.")
            return

        print(f"✅ Found {len(live_games)} live games!")
        print()

        # Display games
        for i, game in enumerate(live_games[:10], 1):  # Show max 10 games
            print(f"{i}. {format_game_info(game)}")
            print()

        # Ask user to select a game
        if len(live_games) == 1:
            selected_game = live_games[0]
            print(f"🎯 Using the only available game: {selected_game.get('match_id')}")
        else:
            while True:
                try:
                    choice = input(
                        f"Select a game (1-{min(len(live_games), 10)}) or 'q' to quit: "
                    ).strip()
                    if choice.lower() == "q":
                        print("👋 Goodbye!")
                        return

                    game_index = int(choice) - 1
                    if 0 <= game_index < min(len(live_games), 10):
                        selected_game = live_games[game_index]
                        break
                    else:
                        print(
                            f"❌ Please enter a number between 1 and {min(len(live_games), 10)}"
                        )
                except ValueError:
                    print("❌ Please enter a valid number or 'q' to quit")

        match_id = selected_game.get("match_id")
        print()
        print(f"🚀 Starting Dota 2 Coach with Match ID: {match_id}")
        print("=" * 50)

        # Import and run the coach
        sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
        from example_dota_coach import main as coach_main

        # Run the coach with the selected game ID
        await coach_main(game_id=str(match_id), interval=2)

    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
