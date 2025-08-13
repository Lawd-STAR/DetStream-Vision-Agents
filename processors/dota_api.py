"""
Valve Dota 2 API Pre-processor Implementation

This module provides integration with Valve's official Dota 2 Web API
to fetch real-time game statistics and player performance data.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
import time
import asyncio

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


def dota_api(
    game_id: str, api_key: Optional[str] = None, player_id: Optional[str] = None
) -> "DotaAPI":
    """
    Factory function for creating Dota API pre-processor.

    Args:
        game_id: The Dota 2 match ID to track
        api_key: Steam API key for Dota 2 API access
        player_id: Steam ID of the player to focus on

    Returns:
        DotaAPI pre-processor instance
    """
    return DotaAPI(game_id, api_key, player_id)


class DotaAPI:
    """
    Valve Dota 2 API integration for real-time game statistics.

    This class fetches and processes live game data from Valve's official
    Dota 2 Web API to provide coaching insights based on actual game performance.

    API Documentation: https://wiki.teamfortress.com/wiki/WebAPI#Dota_2

    Example usage:
        dota = dota_api("7853972481", api_key="your_steam_api_key", player_id="76561198012345678")
        result = dota.process(current_game_time)
    """

    # Valve's Dota 2 API endpoints
    BASE_URL = "https://api.steampowered.com"
    ENDPOINTS = {
        "match_details": "/IDOTA2Match_570/GetMatchDetails/V001/",
        "live_league_games": "/IDOTA2Match_570/GetLiveLeagueGames/v0001/",
        "match_history": "/IDOTA2Match_570/GetMatchHistory/V001/",
        "heroes": "/IEconDOTA2_570/GetHeroes/v0001/",
        "items": "/IEconDOTA2_570/GetGameItems/v0001/",
        "player_summaries": "/ISteamUser/GetPlayerSummaries/v0002/",
    }

    def __init__(
        self,
        game_id: str,
        api_key: Optional[str] = None,
        player_id: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the Valve Dota API pre-processor.

        Args:
            game_id: Dota 2 match ID (can be live match or completed match)
            api_key: Steam API key for accessing Dota 2 API
            player_id: Steam ID (64-bit) of the player to focus on
            **kwargs: Additional configuration options
        """
        self.game_id = str(game_id)
        self.api_key = api_key or os.getenv("STEAM_API_KEY")
        self.player_id = str(player_id) if player_id else None
        self.kwargs = kwargs

        self.logger = logging.getLogger("DotaAPI")

        if not self.api_key:
            self.logger.warning("⚠️ No Steam API key provided. Using mock data.")
            self.use_mock = True
        else:
            self.use_mock = False

        if not AIOHTTP_AVAILABLE:
            self.logger.warning("⚠️ aiohttp not available. Using mock data.")
            self.use_mock = True

        # Cache for API responses
        self._match_cache = {}
        self._heroes_cache = {}
        self._items_cache = {}
        self._last_fetch_time = 0
        self._cache_duration = 30  # seconds

        # Mock game state tracking for fallback
        self._game_start_time = time.time()

        self.logger.info(f"Initialized Dota API processor for match: {game_id}")

    async def process(self, data: Any) -> Dict[str, Any]:
        """
        Process input data and fetch current game statistics from Valve API.

        Args:
            data: Input data (can be game time, player action, etc.)

        Returns:
            Dictionary containing current game statistics and analysis
        """
        try:
            current_time = time.time()

            if self.use_mock:
                return await self._process_mock_data(current_time)

            # Check cache first
            if (
                current_time - self._last_fetch_time < self._cache_duration
                and self.game_id in self._match_cache
            ):
                match_data = self._match_cache[self.game_id]
            else:
                # Fetch fresh data from Valve API
                match_data = await self._fetch_match_details()
                if match_data:
                    self._match_cache[self.game_id] = match_data
                    self._last_fetch_time = current_time
                else:
                    return await self._process_mock_data(current_time)

            # Process the match data
            game_stats = self._process_match_data(match_data)
            player_stats = self._extract_player_stats(match_data)
            match_analysis = self._analyze_performance(game_stats, player_stats)

            return {
                "game_id": self.game_id,
                "game_stats": game_stats,
                "player_stats": player_stats,
                "analysis": match_analysis,
                "recommendations": self._generate_recommendations(match_analysis),
                "timestamp": current_time,
                "data_source": "valve_api",
            }

        except Exception as e:
            self.logger.error(f"Error processing Dota API data: {e}")
            # Fallback to mock data on error
            return await self._process_mock_data(time.time())

    async def _fetch_live_games(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch live league games from Valve's Dota 2 API."""
        url = f"{self.BASE_URL}{self.ENDPOINTS['live_league_games']}"
        params = {"key": self.api_key, "format": "json"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "result" in data and "games" in data["result"]:
                            games = data["result"]["games"]
                            self.logger.info(f"✅ Found {len(games)} live games")
                            return games
                        else:
                            self.logger.warning("⚠️ No live games found")
                            return []
                    else:
                        self.logger.error(
                            f"❌ API request failed with status {response.status}"
                        )
                        return None

        except asyncio.TimeoutError:
            self.logger.error("❌ API request timed out")
            return None
        except Exception as e:
            self.logger.error(f"❌ Error fetching live games: {e}")
            return None

    async def _fetch_match_details(self) -> Optional[Dict[str, Any]]:
        """Fetch match details from Valve's Dota 2 API."""
        url = f"{self.BASE_URL}{self.ENDPOINTS['match_details']}"
        params = {"key": self.api_key, "match_id": self.game_id, "format": "json"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "result" in data and "error" not in data["result"]:
                            self.logger.debug(
                                f"✅ Fetched match details for {self.game_id}"
                            )
                            return data["result"]
                        else:
                            self.logger.warning(
                                f"⚠️ API returned error: {data.get('result', {}).get('error', 'Unknown error')}"
                            )
                            return None
                    else:
                        self.logger.error(
                            f"❌ API request failed with status {response.status}"
                        )
                        return None

        except asyncio.TimeoutError:
            self.logger.error("❌ API request timed out")
            return None
        except Exception as e:
            self.logger.error(f"❌ Error fetching match details: {e}")
            return None

    async def _fetch_heroes_data(self) -> Dict[int, Dict[str, Any]]:
        """Fetch heroes data from Valve API (cached)."""
        if self._heroes_cache:
            return self._heroes_cache

        url = f"{self.BASE_URL}{self.ENDPOINTS['heroes']}"
        params = {"key": self.api_key, "format": "json", "language": "en_us"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "result" in data and "heroes" in data["result"]:
                            heroes_dict = {}
                            for hero in data["result"]["heroes"]:
                                heroes_dict[hero["id"]] = hero
                            self._heroes_cache = heroes_dict
                            return heroes_dict
        except Exception as e:
            self.logger.error(f"Error fetching heroes data: {e}")

        # Fallback hero data
        return self._get_fallback_heroes()

    def _process_match_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw match data into structured game stats."""
        if not match_data:
            return {}

        duration = match_data.get("duration", 0)
        radiant_win = match_data.get("radiant_win", False)

        # Calculate team scores (kills)
        radiant_kills = sum(
            player.get("kills", 0) for player in match_data.get("players", [])[:5]
        )
        dire_kills = sum(
            player.get("kills", 0) for player in match_data.get("players", [])[5:]
        )

        # Determine game phase based on duration
        if duration < 900:  # 15 minutes
            phase = "early_game"
        elif duration < 2400:  # 40 minutes
            phase = "mid_game"
        else:
            phase = "late_game"

        # Extract tower status
        tower_status_radiant = match_data.get("tower_status_radiant", 0)
        tower_status_dire = match_data.get("tower_status_dire", 0)

        # Count remaining towers (11 towers per team max)
        radiant_towers = bin(tower_status_radiant).count("1")
        dire_towers = bin(tower_status_dire).count("1")

        return {
            "duration": duration,
            "phase": phase,
            "radiant_score": radiant_kills,
            "dire_score": dire_kills,
            "radiant_win": radiant_win,
            "game_mode": match_data.get("game_mode", 0),
            "lobby_type": match_data.get("lobby_type", 0),
            "towers_standing": {"radiant": radiant_towers, "dire": dire_towers},
            "barracks_status": {
                "radiant": match_data.get("barracks_status_radiant", 0),
                "dire": match_data.get("barracks_status_dire", 0),
            },
            "first_blood_time": match_data.get("first_blood_time", 0),
            "match_seq_num": match_data.get("match_seq_num", 0),
        }

    def _extract_player_stats(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract player-specific statistics from match data."""
        players = match_data.get("players", [])

        if not players:
            return {}

        # If player_id is specified, find that player
        target_player = None
        if self.player_id:
            for player in players:
                if (
                    str(player.get("account_id", "")) == self.player_id
                    or str(player.get("steamid", "")) == self.player_id
                ):
                    target_player = player
                    break

        # If no specific player found, use first player (radiant carry typically)
        if not target_player:
            target_player = players[0]

        # Extract comprehensive player stats
        stats = {
            "account_id": target_player.get("account_id"),
            "hero_id": target_player.get("hero_id"),
            "player_slot": target_player.get("player_slot"),
            "kills": target_player.get("kills", 0),
            "deaths": target_player.get("deaths", 0),
            "assists": target_player.get("assists", 0),
            "last_hits": target_player.get("last_hits", 0),
            "denies": target_player.get("denies", 0),
            "gold": target_player.get("gold", 0),
            "gold_per_min": target_player.get("gold_per_min", 0),
            "xp_per_min": target_player.get("xp_per_min", 0),
            "level": target_player.get("level", 1),
            "hero_damage": target_player.get("hero_damage", 0),
            "tower_damage": target_player.get("tower_damage", 0),
            "hero_healing": target_player.get("hero_healing", 0),
            "gold_spent": target_player.get("gold_spent", 0),
            "scaled_hero_damage": target_player.get("scaled_hero_damage", 0),
            "scaled_tower_damage": target_player.get("scaled_tower_damage", 0),
            "scaled_hero_healing": target_player.get("scaled_hero_healing", 0),
        }

        # Process items
        items = []
        for i in range(6):  # 6 item slots
            item_id = target_player.get(f"item_{i}", 0)
            if item_id > 0:
                items.append(item_id)

        stats["items"] = items
        stats["backpack"] = [
            target_player.get("backpack_0", 0),
            target_player.get("backpack_1", 0),
            target_player.get("backpack_2", 0),
        ]

        # Determine team and position
        player_slot = target_player.get("player_slot", 0)
        is_radiant = player_slot < 128
        position_slot = player_slot % 5

        position_names = [
            "safe_lane",
            "mid_lane",
            "off_lane",
            "support",
            "hard_support",
        ]
        stats["team"] = "radiant" if is_radiant else "dire"
        stats["position"] = (
            position_names[position_slot] if position_slot < 5 else "unknown"
        )

        return stats

    def _analyze_performance(
        self, game_stats: Dict, player_stats: Dict
    ) -> Dict[str, Any]:
        """Analyze player performance based on Valve API data."""
        if not player_stats or not game_stats:
            return {}

        game_duration_minutes = game_stats.get("duration", 0) / 60

        # Calculate performance metrics
        kills = player_stats.get("kills", 0)
        deaths = player_stats.get("deaths", 1)  # Avoid division by zero
        assists = player_stats.get("assists", 0)
        kda_ratio = (kills + assists) / deaths

        last_hits = player_stats.get("last_hits", 0)
        last_hit_rate = last_hits / max(1, game_duration_minutes)

        gpm = player_stats.get("gold_per_min", 0)
        gpm_rating = (
            "excellent" if gpm > 600 else "good" if gpm > 450 else "needs_improvement"
        )

        xpm = player_stats.get("xp_per_min", 0)
        level = player_stats.get("level", 1)

        # Identify areas for improvement
        issues = []
        if deaths > game_duration_minutes * 0.5:
            issues.append("dying_too_much")
        if last_hit_rate < 5 and player_stats.get("position") in [
            "safe_lane",
            "mid_lane",
        ]:
            issues.append("poor_farming")
        if kda_ratio < 1.0:
            issues.append("low_impact")
        if gpm < 300:
            issues.append("very_low_farm")
        if level < (game_duration_minutes * 0.4):
            issues.append("low_experience")

        # Identify strengths
        strengths = []
        if kda_ratio > 2.0:
            strengths.append("good_kda")
        if last_hit_rate > 8:
            strengths.append("excellent_farming")
        if assists > kills:
            strengths.append("team_player")
        if player_stats.get("hero_damage", 0) > 20000:
            strengths.append("high_damage_output")
        if player_stats.get("tower_damage", 0) > 5000:
            strengths.append("good_objective_focus")

        return {
            "kda_ratio": round(kda_ratio, 2),
            "last_hit_rate": round(last_hit_rate, 1),
            "gpm_rating": gpm_rating,
            "performance_score": self._calculate_performance_score(
                player_stats, game_stats
            ),
            "issues": issues,
            "strengths": strengths,
            "game_impact": self._assess_game_impact(player_stats, game_stats),
            "farming_efficiency": self._assess_farming_efficiency(
                player_stats, game_stats
            ),
            "combat_effectiveness": self._assess_combat_effectiveness(player_stats),
            "objective_contribution": self._assess_objective_contribution(player_stats),
        }

    def _calculate_performance_score(
        self, player_stats: Dict, game_stats: Dict
    ) -> float:
        """Calculate an overall performance score (0-100) based on Valve API data."""
        if not player_stats or not game_stats:
            return 50.0

        score = 50  # Base score

        # KDA contribution (25 points max)
        kda = (player_stats.get("kills", 0) + player_stats.get("assists", 0)) / max(
            1, player_stats.get("deaths", 1)
        )
        score += min(25, kda * 8)

        # GPM contribution (20 points max)
        gpm = player_stats.get("gold_per_min", 0)
        gpm_bonus = (gpm - 300) / 15
        score += min(20, max(-20, gpm_bonus))

        # Last hit contribution (15 points max)
        game_duration_minutes = game_stats.get("duration", 0) / 60
        expected_lh = game_duration_minutes * 6  # 6 LH per minute expected
        lh_ratio = player_stats.get("last_hits", 0) / max(1, expected_lh)
        score += min(15, lh_ratio * 15)

        # Hero damage contribution (15 points max)
        hero_damage = player_stats.get("hero_damage", 0)
        damage_per_min = hero_damage / max(1, game_duration_minutes)
        damage_score = min(15, damage_per_min / 400)  # 400 DPM = 1 point
        score += damage_score

        # Level/XPM contribution (10 points max)
        xpm = player_stats.get("xp_per_min", 0)
        xpm_bonus = (xpm - 400) / 50
        score += min(10, max(-10, xpm_bonus))

        # Tower damage contribution (5 points max)
        tower_damage = player_stats.get("tower_damage", 0)
        tower_score = min(5, tower_damage / 2000)  # 2000 tower damage = 1 point
        score += tower_score

        return max(0, min(100, round(score, 1)))

    def _assess_game_impact(self, player_stats: Dict, game_stats: Dict) -> str:
        """Assess the player's overall impact on the game."""
        kda = (player_stats.get("kills", 0) + player_stats.get("assists", 0)) / max(
            1, player_stats.get("deaths", 1)
        )
        gpm = player_stats.get("gold_per_min", 0)
        hero_damage = player_stats.get("hero_damage", 0)

        if kda > 2.0 and gpm > 500 and hero_damage > 20000:
            return "very_high"
        elif kda > 1.5 and gpm > 400:
            return "high"
        elif kda > 1.0 or gpm > 350:
            return "medium"
        else:
            return "low"

    def _assess_farming_efficiency(self, player_stats: Dict, game_stats: Dict) -> str:
        """Assess farming efficiency."""
        gpm = player_stats.get("gold_per_min", 0)
        last_hits = player_stats.get("last_hits", 0)
        game_duration_minutes = game_stats.get("duration", 0) / 60
        lh_per_min = last_hits / max(1, game_duration_minutes)

        if gpm > 600 and lh_per_min > 8:
            return "excellent"
        elif gpm > 450 and lh_per_min > 6:
            return "good"
        elif gpm > 350 and lh_per_min > 4:
            return "average"
        else:
            return "poor"

    def _assess_combat_effectiveness(self, player_stats: Dict) -> str:
        """Assess combat effectiveness."""
        hero_damage = player_stats.get("hero_damage", 0)
        kills = player_stats.get("kills", 0)
        assists = player_stats.get("assists", 0)
        deaths = player_stats.get("deaths", 1)

        kda = (kills + assists) / deaths

        if hero_damage > 25000 and kda > 3:
            return "excellent"
        elif hero_damage > 15000 and kda > 2:
            return "good"
        elif hero_damage > 8000 and kda > 1:
            return "average"
        else:
            return "poor"

    def _assess_objective_contribution(self, player_stats: Dict) -> str:
        """Assess contribution to objectives."""
        tower_damage = player_stats.get("tower_damage", 0)

        if tower_damage > 8000:
            return "excellent"
        elif tower_damage > 4000:
            return "good"
        elif tower_damage > 2000:
            return "average"
        else:
            return "poor"

    async def _process_mock_data(self, current_time: float) -> Dict[str, Any]:
        """Process mock data when API is unavailable."""
        import random

        game_duration = int(current_time - self._game_start_time)

        # Mock game stats
        game_stats = {
            "duration": game_duration,
            "phase": "early_game"
            if game_duration < 900
            else "mid_game"
            if game_duration < 1800
            else "late_game",
            "radiant_score": max(0, random.randint(0, 30) + (game_duration // 60)),
            "dire_score": max(0, random.randint(0, 30) + (game_duration // 60)),
            "towers_standing": {
                "radiant": random.randint(6, 11),
                "dire": random.randint(6, 11),
            },
            "game_mode": 22,  # Ranked matchmaking
            "lobby_type": 7,  # Ranked matchmaking
        }

        # Mock player stats
        time_factor = game_duration / 60
        base_gpm = 400

        player_stats = {
            "kills": random.randint(0, max(1, int(time_factor * 0.8))),
            "deaths": random.randint(0, max(1, int(time_factor * 0.4))),
            "assists": random.randint(0, max(1, int(time_factor * 1.2))),
            "last_hits": random.randint(0, max(10, int(time_factor * 8))),
            "denies": random.randint(0, max(2, int(time_factor * 1.5))),
            "gold_per_min": base_gpm + random.randint(-100, 200),
            "xp_per_min": 500 + random.randint(-150, 250),
            "level": min(30, max(1, int(time_factor * 0.8) + random.randint(-2, 3))),
            "hero_damage": random.randint(5000, 30000),
            "tower_damage": random.randint(1000, 8000),
            "hero_healing": random.randint(2000, 15000),
            "hero_id": 14,  # Pudge
            "team": "radiant",
            "position": random.choice(
                ["safe_lane", "mid_lane", "off_lane", "support", "hard_support"]
            ),
        }

        match_analysis = self._analyze_performance(game_stats, player_stats)

        return {
            "game_id": self.game_id,
            "game_stats": game_stats,
            "player_stats": player_stats,
            "analysis": match_analysis,
            "recommendations": self._generate_recommendations(match_analysis),
            "timestamp": current_time,
            "data_source": "mock",
        }

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate coaching recommendations based on analysis."""
        recommendations = []

        if not analysis:
            return ["Keep playing and focus on improvement!"]

        issues = analysis.get("issues", [])
        strengths = analysis.get("strengths", [])

        # Issue-based recommendations
        if "dying_too_much" in issues:
            recommendations.append(
                "🛡️ Focus on positioning - you're dying too frequently. Play safer and watch the minimap!"
            )

        if "poor_farming" in issues:
            recommendations.append(
                "🌾 Work on your last hitting! Aim for at least 6-8 last hits per minute."
            )

        if "low_impact" in issues:
            recommendations.append(
                "⚔️ Try to be more active in fights and help your team more."
            )

        if "very_low_farm" in issues:
            recommendations.append(
                "💰 Your farm is critically low. Focus on efficient farming patterns and map control."
            )

        if "low_experience" in issues:
            recommendations.append(
                "📈 You're falling behind in levels. Stay in XP range of creeps and participate in kills."
            )

        # Strength-based positive reinforcement
        if "good_kda" in strengths:
            recommendations.append(
                "🏆 Great KDA! You're making smart decisions in fights."
            )

        if "excellent_farming" in strengths:
            recommendations.append(
                "💎 Excellent farming efficiency! Keep up the good work."
            )

        if "team_player" in strengths:
            recommendations.append(
                "🤝 Great team play! Your assist count shows you're helping your teammates."
            )

        if "high_damage_output" in strengths:
            recommendations.append(
                "💥 Impressive damage output! You're making a real impact in fights."
            )

        if "good_objective_focus" in strengths:
            recommendations.append(
                "🏗️ Good objective focus! Tower damage is crucial for winning games."
            )

        # Performance-based recommendations
        performance_score = analysis.get("performance_score", 50)
        if performance_score > 80:
            recommendations.append(
                "🌟 Outstanding performance! You're dominating this match!"
            )
        elif performance_score > 60:
            recommendations.append("✨ Solid performance! Keep up the momentum!")
        elif performance_score < 40:
            recommendations.append(
                "📚 Focus on the fundamentals - farming, positioning, and map awareness."
            )

        if not recommendations:
            recommendations.append(
                "Keep up the solid performance! Look for opportunities to help your team."
            )

        return recommendations

    def _get_fallback_heroes(self) -> Dict[int, Dict[str, Any]]:
        """Fallback hero data when API is unavailable."""
        return {
            14: {"name": "pudge", "localized_name": "Pudge"},
            74: {"name": "invoker", "localized_name": "Invoker"},
            25: {"name": "crystal_maiden", "localized_name": "Crystal Maiden"},
            2: {"name": "axe", "localized_name": "Axe"},
            6: {"name": "drow_ranger", "localized_name": "Drow Ranger"},
        }

    def __repr__(self) -> str:
        """String representation of the processor."""
        return f"DotaAPI(game_id='{self.game_id}', player_id='{self.player_id}', use_mock={self.use_mock})"
