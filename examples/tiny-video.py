#!/usr/bin/env python3
"""
Tiny Video Example - Minimal Stream video call with frame logging

This is the smallest possible example that:
1. Opens a Stream video call
2. Logs when video is received using track.recv()

Usage:
    python tiny-video.py
"""

import asyncio
import logging
import os
import traceback
from uuid import uuid4

from dotenv import load_dotenv
from getstream.stream import Stream
from getstream.models import UserRequest
from getstream.video import rtc
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig, TrackType

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def on_track_added(track_id, track_type, user, ai_connection):
    """Handle a new track being added to the ai connection."""
    logger.info(f"Track added: {track_id} for user {user.user_id} of type {track_type}")
    
    if track_type != "video":
        return

    track = ai_connection.subscriber_pc.add_track_subscriber(track_id)

    if track:
        logger.info(f"✅ Video track found: {track_id}")
        frame_count = 0
        
        while True:
            try:
                video_frame = await track.recv()
                if video_frame:
                    frame_count += 1
                    logger.info(f"📹 Received video frame #{frame_count}: {video_frame.width}x{video_frame.height}")
                    
            except Exception as e:
                logger.error(f"Error receiving track: {e}")
                break
    else:
        logger.error(f"❌ Track not found: {track_id}")


async def main():
    logger.info("🎥 Tiny Video Example")
    logger.info("=" * 30)

    # Load environment variables
    load_dotenv()

    # Initialize Stream client
    client = Stream.from_env()

    # Create a unique call
    call_id = f"tiny-video-{str(uuid4())}"
    call = client.video.call("default", call_id)
    logger.info(f"📞 Call ID: {call_id}")

    # Create users
    human_user_id = f"human-{str(uuid4())[:8]}"
    ai_user_id = f"ai-{str(uuid4())[:8]}"

    # Create users in Stream
    client.upsert_users(
        UserRequest(id=human_user_id, name="Human User"),
        UserRequest(id=ai_user_id, name="AI Observer")
    )

    # Create token for browser access
    token = client.create_token(human_user_id, expiration=3600)

    # Create the call
    call.get_or_create(data={"created_by_id": "tiny-video-example"})

    try:
        # Join as AI observer
        async with await rtc.join(
            call,
            ai_user_id,
            subscription_config=SubscriptionConfig(
                default=TrackSubscriptionConfig(track_types=[TrackType.TRACK_TYPE_VIDEO])
            )
        ) as ai_connection:
            
            # Set up track added handler
            ai_connection.on(
                "track_added",
                lambda track_id, track_type, user: asyncio.create_task(
                    on_track_added(track_id, track_type, user, ai_connection)
                )
            )

            # Open browser for human user
            base_url = f"{os.getenv('EXAMPLE_BASE_URL', 'https://pronto-staging.getstream.io')}/join/"
            params = {
                "api_key": client.api_key,
                "token": token,
                "skip_lobby": "true",
                "video_encoder": "vp8",
            }
            url = f"{base_url}{call_id}?api_key={client.api_key}&token={token}&skip_lobby=true&video_encoder=vp8"
            
            logger.info(f"🌐 Open this URL in your browser: {url}")
            logger.info("📹 Start your camera and you'll see frame logs here!")

            # Wait for the connection
            await ai_connection.wait()
            
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(f"Stacktrace: {traceback.format_exc()}")
    finally:
        # Clean up
        logger.info("🧹 Cleaning up...")
        client.delete_users([human_user_id, ai_user_id])


if __name__ == "__main__":
    asyncio.run(main())
