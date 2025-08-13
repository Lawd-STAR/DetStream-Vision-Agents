#!/usr/bin/env python3
"""
Blue Video Transformer Example for Stream Agents

This example demonstrates how to use the Agent class with a video transformer
that applies a blue tint to video frames from a Stream video call.

Usage:
    python example_blue_video_transformer.py [--interval SECONDS] [--output DIR]

The example will:
1. Create a Stream video call
2. Join as an AI Agent with video transformation enabled
3. Apply a blue tint to all video frames
4. Optionally capture the transformed frames at specified intervals
5. Display the captured frames count and file names

Requirements:
    - STREAM_API_KEY and STREAM_SECRET environment variables
    - getstream-python package
    - Pillow (PIL) for image processing
    - aiortc for video frame handling
    - numpy for image manipulation
"""

import argparse
import asyncio
import logging
import sys
import traceback
from pathlib import Path
from uuid import uuid4

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
import colorsys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from getstream.stream import Stream

# Import from the top-level utils.py file (not the utils package)
import importlib.util
spec = importlib.util.spec_from_file_location("utils", str(Path(__file__).parent.parent / "utils.py"))
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

# Import Agent base class
sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))
from agents import Agent, VideoTransformer, ImageProcessor

# Configure logging with VP8 decoder warning filter
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Filter out VP8 decoder warnings which are normal during video startup
class VP8WarningFilter(logging.Filter):
    def filter(self, record):
        return not ("Vp8Decoder() failed to decode" in record.getMessage())

# Apply filter to reduce noise
logging.getLogger().addFilter(VP8WarningFilter())
logger = logging.getLogger(__name__)


class BlueVideoTransformer(VideoTransformer):
    """Transforms video frames to have a blue tint."""

    def __init__(self, blue_intensity: float = 0.5):
        """
        Initialize the blue video transformer.

        Args:
            blue_intensity: Intensity of the blue tint (0.0 to 1.0)
        """
        self.blue_intensity = max(0.0, min(1.0, blue_intensity))
        logger.info(f"🔵 Blue transformer initialized with intensity: {self.blue_intensity}")

    async def transform_frame(self, frame: Image.Image) -> Image.Image:
        """
        Transform a video frame to have a subtle blue tint overlay.

        Args:
            frame: PIL Image to transform

        Returns:
            PIL Image with blue tint applied
        """
        try:
            # Ensure we have a valid RGB image
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            
            # Convert PIL Image to numpy array
            img_array = np.array(frame, dtype=np.float32)
            
            # Validate image dimensions
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                logger.warning("Invalid image format, returning original")
                return frame
            
            # Start with original image to preserve content
            result = img_array.copy()
            
            # More noticeable but still natural blue enhancement
            # Method 1: Moderate boost to blue channel (up to 8% max)
            result[:, :, 2] *= (1.0 + self.blue_intensity * 0.08)
            
            # Method 2: Add noticeable blue cast (max 12 RGB units)
            blue_cast = self.blue_intensity * 12
            result[:, :, 2] += blue_cast
            
            # Method 3: Reduce red/warm tones more noticeably (up to 4% reduction)
            result[:, :, 0] *= (1.0 - self.blue_intensity * 0.04)
            result[:, :, 1] *= (1.0 - self.blue_intensity * 0.02)  # Slight green reduction too
            
            # Ensure values stay within valid range [0, 255] and convert to uint8
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            # Convert back to PIL Image with explicit RGB mode
            transformed_frame = Image.fromarray(result, mode='RGB')
            
            return transformed_frame
            
        except Exception as e:
            logger.error(f"❌ Error applying blue transformation: {e}")
            # Return original frame if transformation fails
            return frame


class VideoFrameCapture(ImageProcessor):
    """Handles video frame capture and storage."""

    def __init__(self, output_dir: str = "blue_frames", capture_interval: int = 5):
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
        logger.info(f"📁 Saving blue-transformed frames to: {self.output_dir.absolute()}")

    async def should_capture_frame(self) -> bool:
        """Check if it's time to capture a new frame."""
        current_time = asyncio.get_event_loop().time()

        if current_time - self.last_capture_time >= self.capture_interval:
            self.last_capture_time = current_time
            return True
        return False

    async def process_image(
        self, image: Image.Image, user_id: str, metadata: dict = None
    ) -> None:
        """Process image by saving it as JPG."""
        if await self.should_capture_frame():
            await self.capture_frame(image, user_id)

    async def capture_frame(self, frame: Image.Image, user_id: str) -> str:
        """
        Capture and save a blue-transformed video frame as JPG.

        Args:
            frame: PIL Image to save
            user_id: ID of the user whose video is being captured

        Returns:
            Path to the saved image file
        """
        timestamp = int(asyncio.get_event_loop().time())
        filename = f"blue_frame_{user_id}_{timestamp}_{self.frame_count:04d}.jpg"
        filepath = self.output_dir / filename

        # Save the frame as JPG
        frame.save(filepath, "JPEG", quality=90)

        self.frame_count += 1
        logger.info(
            f"🔵 Captured blue frame {self.frame_count}: {filename} ({frame.width}x{frame.height})"
        )

        return str(filepath)


def create_user(client: Stream, user_id: str, name: str):
    """Create a user in Stream."""
    try:
        from getstream.models import UserRequest

        user_request = UserRequest(id=user_id, name=name)
        client.upsert_users(user_request)
        logger.info(f"👤 Created user: {name} ({user_id})")
    except Exception as e:
        logger.error(f"❌ Failed to create user {user_id}: {e}")


async def main(interval: int = 5, output_dir: str = "blue_frames", capture_frames: bool = True):
    """Main function to run the blue video transformer example."""

    print("🔵 Stream Agents - Blue Video Transformer Example")
    print("=" * 60)
    print("This example applies a blue tint to video frames from a Stream video call.")
    if capture_frames:
        print(f"Transformed frames will be captured every {interval} seconds.")
    print("Join the call from your browser to see your video being transformed!")
    print()

    # Load environment variables
    load_dotenv()

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
    call_id = f"blue-transform-{str(uuid4())[:8]}"
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

    # Initialize Blue Video Transformer with more noticeable intensity
    blue_transformer = BlueVideoTransformer(blue_intensity=0.7)

    # Initialize optional frame capture
    image_processors = []
    frame_capture = None
    if capture_frames:
        frame_capture = VideoFrameCapture(output_dir=output_dir, capture_interval=interval)
        image_processors = [frame_capture]

    # Create Agent with blue video transformer
    agent = Agent(
        instructions="I am a blue video transformer bot that applies a blue tint to video frames.",
        video_transformer=blue_transformer,  # Use our blue transformer
        image_interval=interval if capture_frames else None,
        image_processors=image_processors,
        target_user_id=None,  # Transform video from any user
        name="Blue Video Transformer Bot",
    )

    # User creation callback
    def create_agent_user(bot_id: str, name: str):
        create_user(client, bot_id, name)

    try:
        # Open browser for participant to join
        utils.open_pronto(client.api_key, participant_token, call_id)

        print()
        print("🎯 Ready to transform video frames to blue!")
        print("   • Join the call from your browser")
        print("   • Turn on your camera")
        print("   • Your video will appear with a blue tint")
        if capture_frames:
            print(f"   • Blue frames will be captured every {interval} seconds")
        print("   • Press Ctrl+C to stop")
        print()

        # Join the call with the Agent
        await agent.join(call, user_creation_callback=create_agent_user)

    except KeyboardInterrupt:
        logger.info("⏹️  Blue video transformer stopped by user")
    except Exception as e:
        logger.error(f"❌ Error during blue video transformation: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Cleanup: Delete created users
        try:
            logger.info("🧹 Cleaning up users...")
            client.delete_users([agent.bot_id, participant_user_id])
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

        # Display summary
        print()
        print("📊 Transformation Summary:")
        if frame_capture:
            print(f"   • Total blue frames captured: {frame_capture.frame_count}")
            print(f"   • Frames saved to: {frame_capture.output_dir.absolute()}")

            if frame_capture.frame_count > 0:
                print("   • Captured files:")
                for jpg_file in sorted(frame_capture.output_dir.glob("*.jpg")):
                    print(f"     - {jpg_file.name}")
        else:
            print("   • Video transformation applied in real-time")
        print("   • Blue tint intensity: 70% (noticeable but natural)")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Blue Video Transformer Example for Stream Agents"
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=5,
        help="Capture interval in seconds (default: 5)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="blue_frames",
        help="Output directory for captured frames (default: blue_frames)",
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="Disable frame capture, only apply transformation",
    )

    args = parser.parse_args()

    # Run the example with parsed arguments
    asyncio.run(main(
        interval=args.interval, 
        output_dir=args.output, 
        capture_frames=not args.no_capture
    ))
