#!/usr/bin/env python3
"""
YOLO Pose Processor Example for Stream Agents

This example demonstrates how to use the YOLOPoseProcessor with the Agent class
to perform real-time pose detection on video frames from a Stream video call.

Usage:
    python example_yolo_pose_processor.py [--interval SECONDS] [--output DIR]

The example will:
1. Create a Stream video call
2. Join as an AI Agent with YOLO pose detection enabled
3. Apply pose detection overlays to all video frames
4. Publish the pose-annotated video back to the call
5. Optionally capture the annotated frames at specified intervals

Requirements:
    - STREAM_API_KEY and STREAM_SECRET environment variables
    - getstream-python package
    - ultralytics package (for YOLO)
    - OpenCV (cv2)
    - YOLO pose model file (yolo11n-pose.pt)
"""

import argparse
import asyncio
import logging
import sys
import traceback
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import open_demo
from getstream.stream import Stream

# Import Agent base class and processor mixins
from agents.agents2 import Agent

# Import processor base classes and mixins
from processors.yolo_pose_processor import YOLOPoseProcessor
from processors.base_processor import ImageCapture

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
    interval: int = 5, output_dir: str = "pose_frames", capture_frames: bool = True
):
    """Main function to run the YOLO pose processor example."""

    print("🤖 Stream Agents - YOLO Pose Detection Example")
    print("=" * 60)
    print(
        "This example applies YOLO pose detection to video frames from a Stream video call."
    )
    if capture_frames:
        print(f"Pose-annotated frames will be captured every {interval} seconds.")
    print("Join the call from your browser to see your pose being detected!")
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
    call_id = f"pose-detection-{str(uuid4())[:8]}"
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

    # Initialize YOLO Pose Processor
    try:
        pose_processor = YOLOPoseProcessor(
            model_path="yolo11n-pose.pt",  # Will download if not exists
            conf_threshold=0.5,
            imgsz=512,
            device="cpu",  # Use "cuda" if you have GPU
            interval=0,  # Process every frame
            enable_hand_tracking=True,
            enable_wrist_highlights=True,
        )
        logger.info("✅ YOLO Pose Processor initialized")
    except ImportError as e:
        logger.error(f"❌ Failed to initialize YOLO Pose Processor: {e}")
        logger.error("Please install ultralytics: pip install ultralytics")
        return
    except Exception as e:
        logger.error(f"❌ Failed to initialize YOLO Pose Processor: {e}")
        return

    # Initialize processors list
    processors = [pose_processor]
    frame_capture = None
    if capture_frames:
        frame_capture = ImageCapture(output_dir=output_dir, interval=interval)
        processors.append(frame_capture)

    # Create Agent with YOLO pose processor
    agent = Agent(
        processors=processors,  # Use our processors list
    )

    # Create the agent user before joining
    create_user(client, agent.agent_user.id, "YOLO Pose Detection Bot")

    # Open demo with browser for human user
    open_demo(client, call.id)

    try:
        print()
        print("🎯 Ready to detect poses in video frames!")
        print("   • Join the call from your browser")
        print("   • Turn on your camera")
        print("   • Your video will appear with pose detection overlays")
        if capture_frames:
            print(f"   • Pose frames will be captured every {interval} seconds")
        print("   • Press Ctrl+C to stop")
        print()

        # Join the call with the Agent
        await agent.join(call)

    except KeyboardInterrupt:
        logger.info("⏹️  YOLO pose detection stopped by user")
    except Exception as e:
        logger.error(f"❌ Error during YOLO pose detection: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Cleanup: Delete created users
        try:
            logger.info("🧹 Cleaning up users...")
            client.delete_users([agent.agent_user.id, participant_user_id])
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

        # Cleanup pose processor
        try:
            pose_processor.cleanup()
        except Exception as e:
            logger.error(f"❌ Error during pose processor cleanup: {e}")

        # Display summary
        print()
        print("📊 Pose Detection Summary:")
        if frame_capture:
            print(f"   • Total pose frames captured: {frame_capture.frame_count}")
            print(f"   • Frames saved to: {frame_capture.output_dir.absolute()}")

            if frame_capture.frame_count > 0:
                print("   • Captured files:")
                for jpg_file in sorted(frame_capture.output_dir.glob("*.jpg")):
                    print(f"     - {jpg_file.name}")
        else:
            print("   • Pose detection applied in real-time")
        print("   • YOLO model: yolo11n-pose.pt")
        print("   • Confidence threshold: 50%")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="YOLO Pose Detection Example for Stream Agents"
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
        default="pose_frames",
        help="Output directory for captured frames (default: pose_frames)",
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="Disable frame capture, only apply pose detection",
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
