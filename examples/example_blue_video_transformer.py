#!/usr/bin/env python3
"""
Blue Video Transformer Example for Stream Agents

This example demonstrates how to use the Agent class with a video processor
that applies a blue tint to video frames from a Stream video call and publishes them.

Usage:
    python example_blue_video_transformer.py [--interval SECONDS] [--output DIR]

The example will:
1. Create a Stream video call
2. Join as an AI Agent with video processing enabled
3. Apply a blue tint to all video frames
4. Publish the transformed video back to the call
5. Optionally capture the transformed frames at specified intervals
6. Display the captured frames count and file names

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
from PIL import Image
import aiortc

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import open_demo

from getstream.stream import Stream

# Helper functions now imported from utils module

# Import Agent base class and processor mixins
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.agents2 import Agent

# Import processor base classes and mixins
from processors.base_processor import (
    AudioVideoProcessor,
    ImageProcessorMixin,
    VideoPublisherMixin,
    VideoProcessorMixin,
)

# Configure logging with VP8 decoder warning filter
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Filter out VP8 decoder warnings which are normal during video startup
class VP8WarningFilter(logging.Filter):
    def filter(self, record):
        return "Vp8Decoder() failed to decode" not in record.getMessage()


# Apply filter to reduce noise
logging.getLogger().addFilter(VP8WarningFilter())
logger = logging.getLogger(__name__)


class BlueVideoProcessor(AudioVideoProcessor, VideoProcessorMixin, VideoPublisherMixin):
    """Processes video frames to apply a blue tint and publishes the transformed video."""

    def __init__(self, blue_intensity: float = 0.5, interval: int = 0, *args, **kwargs):
        """
        Initialize the blue video processor.

        Args:
            blue_intensity: Intensity of the blue tint (0.0 to 1.0)
            interval: Processing interval in seconds (0 for every frame)
        """
        super().__init__(
            interval=interval, receive_audio=False, receive_video=True, *args, **kwargs
        )
        self.blue_intensity = max(0.0, min(1.0, blue_intensity))
        self._video_track = None
        logger.info(
            f"🔵 Blue processor initialized with intensity: {self.blue_intensity}"
        )

    def create_video_track(self) -> aiortc.VideoStreamTrack:
        """Create a video track for publishing transformed frames."""
        from agents.agents import TransformedVideoTrack

        self._video_track = TransformedVideoTrack()
        logger.info("🎥 Blue video track created for publishing")
        return self._video_track

    async def process_video(
        self, track: aiortc.mediastreams.MediaStreamTrack, user_id: str
    ):
        """Process video frames from the input track and publish transformed frames."""
        logger.info(f"🔵 Starting blue video processing for user {user_id}")

        try:
            while True:
                try:
                    # Receive video frame from input track
                    video_frame = await track.recv()
                    if not video_frame:
                        continue

                    # Convert to PIL Image
                    img = video_frame.to_image()

                    # Apply blue transformation
                    transformed_img = await self.transform_frame(img)

                    # Publish transformed frame to output track
                    if self._video_track:
                        await self._video_track.add_frame(transformed_img)

                except Exception as e:
                    if "Connection closed" in str(e) or "Track ended" in str(e):
                        logger.info(f"🔌 Video track ended for user {user_id}")
                        break
                    else:
                        logger.error(f"❌ Error processing video frame: {e}")
                        await asyncio.sleep(0.1)  # Brief pause before retry

        except Exception as e:
            logger.error(f"❌ Fatal error in blue video processing: {e}")

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
            if frame.mode != "RGB":
                frame = frame.convert("RGB")

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
            result[:, :, 2] *= 1.0 + self.blue_intensity * 0.08

            # Method 2: Add noticeable blue cast (max 12 RGB units)
            blue_cast = self.blue_intensity * 12
            result[:, :, 2] += blue_cast

            # Method 3: Reduce red/warm tones more noticeably (up to 4% reduction)
            result[:, :, 0] *= 1.0 - self.blue_intensity * 0.04
            result[:, :, 1] *= (
                1.0 - self.blue_intensity * 0.02
            )  # Slight green reduction too

            # Ensure values stay within valid range [0, 255] and convert to uint8
            result = np.clip(result, 0, 255).astype(np.uint8)

            # Convert back to PIL Image with explicit RGB mode
            transformed_frame = Image.fromarray(result, mode="RGB")

            return transformed_frame

        except Exception as e:
            logger.error(f"❌ Error applying blue transformation: {e}")
            # Return original frame if transformation fails
            return frame


class VideoFrameCapture(AudioVideoProcessor, ImageProcessorMixin):
    """Handles video frame capture and storage."""

    def __init__(
        self,
        output_dir: str = "blue_frames",
        capture_interval: int = 5,
        *args,
        **kwargs,
    ):
        """
        Initialize the video frame capture.

        Args:
            output_dir: Directory to save captured frames
            capture_interval: Interval in seconds between captures
        """
        super().__init__(
            interval=capture_interval,
            receive_audio=False,
            receive_video=True,
            *args,
            **kwargs,
        )
        self.output_dir = Path(output_dir)
        self.frame_count = 0

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        logger.info(
            f"📁 Saving blue-transformed frames to: {self.output_dir.absolute()}"
        )

    async def process_image(
        self, image: Image.Image, user_id: str, metadata: dict = None
    ) -> None:
        """Process image by saving it as JPG."""
        if self.should_process():
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


async def main(
    interval: int = 5, output_dir: str = "blue_frames", capture_frames: bool = True
):
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

    # Initialize Blue Video Processor with more noticeable intensity
    blue_processor = BlueVideoProcessor(blue_intensity=0.7, interval=0)

    # Initialize optional frame capture
    processors = [blue_processor]
    frame_capture = None
    if capture_frames:
        frame_capture = VideoFrameCapture(
            output_dir=output_dir, capture_interval=interval
        )
        processors.append(frame_capture)

    # Create Agent with blue video processor
    agent = Agent(
        processors=processors,  # Use our processors list
    )

    # Create the agent user before joining
    create_user(client, agent.agent_user.id, "Blue Video Transformer Bot")

    # Open demo with browser for human user
    open_demo(client, call.id)

    try:
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
        await agent.join(call)

    except KeyboardInterrupt:
        logger.info("⏹️  Blue video transformer stopped by user")
    except Exception as e:
        logger.error(f"❌ Error during blue video transformation: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Cleanup: Delete created users
        try:
            logger.info("🧹 Cleaning up users...")
            client.delete_users([agent.agent_user.id, participant_user_id])
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
    asyncio.run(
        main(
            interval=args.interval,
            output_dir=args.output,
            capture_frames=not args.no_capture,
        )
    )
