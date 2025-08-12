#!/usr/bin/env python3
"""
Video Capture Example for Stream Agents

This example demonstrates how to capture video frames from a Stream video call
and save them as JPG images every 5 seconds. It's inspired by the workout assistant
but simplified to focus on basic video frame capture.

Usage:
    python example_video_capture.py

The example will:
1. Create a Stream video call
2. Join as an AI bot
3. Listen for video tracks from participants
4. Capture frames every 5 seconds and save them as JPG files
5. Display the captured frames count and file names

Requirements:
    - STREAM_API_KEY and STREAM_SECRET environment variables
    - getstream-python package
    - Pillow (PIL) for image processing
    - aiortc for video frame handling
"""

import argparse
import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path
from uuid import uuid4

import aiortc
from dotenv import load_dotenv
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from getstream.stream import Stream
from getstream.video import rtc
from getstream.video.call import Call
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig, TrackType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoFrameCapture:
    """Handles video frame capture and storage."""
    
    def __init__(self, output_dir: str = "captured_frames", capture_interval: int = 5):
        """
        Initialize the video frame capture.
        
        Args:
            output_dir: Directory to save captured frames
            capture_interval: Interval in seconds between captures
        """
        self.output_dir = Path(output_dir)
        self.capture_interval = capture_interval
        self.frame_count = 0
        self.last_capture_time = 0
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"📁 Saving captured frames to: {self.output_dir.absolute()}")
    
    async def should_capture_frame(self) -> bool:
        """Check if it's time to capture a new frame."""
        current_time = asyncio.get_event_loop().time()
        
        if current_time - self.last_capture_time >= self.capture_interval:
            self.last_capture_time = current_time
            return True
        return False
    
    async def capture_frame(self, frame: Image.Image, user_id: str) -> str:
        """
        Capture and save a video frame as JPG.
        
        Args:
            frame: PIL Image to save
            user_id: ID of the user whose video is being captured
            
        Returns:
            Path to the saved image file
        """
        timestamp = int(asyncio.get_event_loop().time())
        filename = f"frame_{user_id}_{timestamp}_{self.frame_count:04d}.jpg"
        filepath = self.output_dir / filename
        
        # Save the frame as JPG
        frame.save(filepath, "JPEG", quality=90)
        
        self.frame_count += 1
        logger.info(f"📸 Captured frame {self.frame_count}: {filename} ({frame.width}x{frame.height})")
        
        return str(filepath)


async def process_video_track(track_id: str, track_type: str, user, target_user_id: str, 
                            ai_connection, frame_capture: VideoFrameCapture):
    """
    Process video frames from a specific track.
    
    Args:
        track_id: ID of the video track
        track_type: Type of track (should be "video")
        user: User object who owns the track
        target_user_id: ID of user we want to capture (None for any user)
        ai_connection: AI bot's RTC connection
        frame_capture: VideoFrameCapture instance
    """
    logger.info(f"🎥 Processing video track: {track_id} from user {user.user_id} (type: {track_type})")
    
    # Only process video tracks
    if track_type != "video":
        logger.debug(f"Ignoring non-video track: {track_type}")
        return
    
    # If target_user_id is specified, only process that user's video
    if target_user_id and user.user_id != target_user_id:
        logger.debug(f"Ignoring video from user {user.user_id} (target: {target_user_id})")
        return
    
    # Subscribe to the video track
    track = ai_connection.subscriber_pc.add_track_subscriber(track_id)
    if not track:
        logger.error(f"❌ Failed to subscribe to track: {track_id}")
        return
    
    logger.info(f"✅ Successfully subscribed to video track from {user.user_id}")
    
    try:
        while True:
            try:
                # Receive video frame
                video_frame: aiortc.mediastreams.VideoFrame = await track.recv()
                if not video_frame:
                    continue
                
                # Check if we should capture this frame
                if await frame_capture.should_capture_frame():
                    # Convert to PIL Image
                    img = video_frame.to_image()
                    
                    # Capture and save the frame
                    await frame_capture.capture_frame(img, user.user_id)
                    
            except Exception as e:
                if "Connection closed" in str(e) or "Track ended" in str(e):
                    logger.info(f"🔌 Video track ended for user {user.user_id}")
                    break
                else:
                    logger.error(f"❌ Error processing video frame: {e}")
                    await asyncio.sleep(1)  # Brief pause before retry
                    
    except Exception as e:
        logger.error(f"❌ Fatal error in video processing: {e}")
        logger.error(traceback.format_exc())


def create_user(client: Stream, user_id: str, name: str):
    """Create a user in Stream."""
    try:
        client.upsert_users([{
            "id": user_id,
            "name": name,
            "role": "user"
        }])
        logger.info(f"👤 Created user: {name} ({user_id})")
    except Exception as e:
        logger.error(f"❌ Failed to create user {user_id}: {e}")


def open_browser(api_key: str, token: str, call_id: str):
    """Open browser with the video call URL."""
    import webbrowser
    from urllib.parse import urlencode
    
    # Use the same URL pattern as the working workout assistant example
    base_url = f"{os.getenv('EXAMPLE_BASE_URL', 'https://pronto-staging.getstream.io')}/join/"
    params = {"api_key": api_key, "token": token, "skip_lobby": "true", "video_encoder": "vp8"}
    
    url = f"{base_url}{call_id}?{urlencode(params)}"
    logger.info(f"🌐 Opening browser: {url}")
    
    try:
        webbrowser.open(url)
        logger.info("✅ Browser opened successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to open browser: {e}")
        logger.info(f"Please manually open this URL: {url}")


async def main():
    """Main function to run the video capture example."""
    
    print("🎥 Stream Agents - Video Capture Example")
    print("=" * 50)
    print("This example captures video frames every 5 seconds from a Stream video call.")
    print("Join the call from your browser to see your video being captured!")
    print()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize Stream client
    try:
        client = Stream.from_env()
        logger.info("✅ Stream client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Stream client: {e}")
        logger.error("Make sure STREAM_API_KEY and STREAM_SECRET are set in your .env file")
        return
    
    # Create a unique call
    call_id = f"video-capture-{str(uuid4())[:8]}"
    call = client.video.call("default", call_id)
    logger.info(f"📞 Call ID: {call_id}")
    
    # Create user IDs
    ai_user_id = f"ai-capture-{str(uuid4())[:8]}"
    participant_user_id = f"participant-{str(uuid4())[:8]}"
    
    # Create users
    create_user(client, ai_user_id, "Video Capture Bot")
    create_user(client, participant_user_id, "Participant")
    
    # Create tokens
    participant_token = client.create_token(participant_user_id, expiration=3600)
    
    # Create the call
    try:
        call.get_or_create(data={"created_by_id": ai_user_id})
        logger.info("✅ Call created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create call: {e}")
        return
    
    # Initialize frame capture
    frame_capture = VideoFrameCapture(
        output_dir="captured_frames",
        capture_interval=5  # 5 seconds
    )
    
    try:
        # Join as AI bot with video subscription
        logger.info(f"🤖 AI bot joining call as: {ai_user_id}")
        
        async with await rtc.join(
            call,
            ai_user_id,
            subscription_config=SubscriptionConfig(
                default=TrackSubscriptionConfig(track_types=[
                    TrackType.TRACK_TYPE_VIDEO,
                ])
            )
        ) as ai_connection:
            
            logger.info("✅ AI bot successfully joined the call")
            
            # Set up event handler for new video tracks
            def on_track_added(track_id, track_type, user):
                logger.info(f"🎬 New track detected: {track_id} ({track_type}) from {user.user_id}")
                asyncio.create_task(
                    process_video_track(
                        track_id, track_type, user, None,  # None means capture from any user
                        ai_connection, frame_capture
                    )
                )
            
            ai_connection.on("track_added", on_track_added)
            
            # Open browser for participant to join
            open_browser(client.api_key, participant_token, call_id)
            
            print()
            print("🎯 Ready to capture video frames!")
            print("   • Join the call from your browser")
            print("   • Turn on your camera")
            print("   • Frames will be captured every 5 seconds")
            print("   • Press Ctrl+C to stop")
            print()
            
            # Wait for the connection
            await ai_connection.wait()
            
    except KeyboardInterrupt:
        logger.info("⏹️  Video capture stopped by user")
    except Exception as e:
        logger.error(f"❌ Error during video capture: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Cleanup: Delete created users
        try:
            logger.info("🧹 Cleaning up users...")
            client.delete_users([ai_user_id, participant_user_id])
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
        
        # Display summary
        print()
        print("📊 Capture Summary:")
        print(f"   • Total frames captured: {frame_capture.frame_count}")
        print(f"   • Frames saved to: {frame_capture.output_dir.absolute()}")
        
        if frame_capture.frame_count > 0:
            print("   • Captured files:")
            for jpg_file in sorted(frame_capture.output_dir.glob("*.jpg")):
                print(f"     - {jpg_file.name}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Video Capture Example for Stream Agents")
    parser.add_argument(
        "-i", "--interval",
        type=int,
        default=5,
        help="Capture interval in seconds (default: 5)"
    )
    parser.add_argument(
        "-o", "--output",
        default="captured_frames",
        help="Output directory for captured frames (default: captured_frames)"
    )
    
    args = parser.parse_args()
    
    # Run the example
    asyncio.run(main())
