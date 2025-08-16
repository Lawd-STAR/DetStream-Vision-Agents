import asyncio
import logging
from pathlib import Path
from typing import Protocol, Any

import aiortc
from PIL.Image import Image

'''
TODO:
- nice syntax for audio/video/image booleans. Mixins?
- cleanup audio_video_image processors in base
- properly forward to image processors from agent (easy)
- figure out aysncio flow for video track recv() loop

'''

class BaseProcessor(Protocol):
    pass

class IntervalProcessor(BaseProcessor):
    # TODO: add interval loop
    pass

class AudioVideoProcessor(BaseProcessor):
    subscribe_audio = True
    subscribe_video = True # don't love how we have both the function def and the variable. can be better.
    subscribe_image = True
    publish_audio = True
    publish_video = True

    def __init__(self, interval: int = 3, receive_audio: bool = False, receive_video: bool = True, *args, **kwargs):
        self.interval = interval
        self.last_process_time = 0

        if self.publish_audio:
            self.audio_track = self.create_audio_track()

        if self.publish_video:
            self.video_track = self.create_video_track()

    def state(self):
        # Returns relevant data for the conversation with the LLM
        pass

    def create_audio_track(self):
        return aiortc.AudioStreamTrack(framerate=24000, stereo=False, format="s16")

    def create_video_track(self):
        return aiortc.VideoStreamTrack()

    async def process_video(self, track: aiortc.mediastreams.MediaStreamTrack, user_id: str, metadata: dict=None):
        pass

    async def process_image(self,  image: Image.Image, user_id: str, metadata: dict=None):
        pass

    async def process_audio(self, audio_data: bytes, user_id: str, metadata: dict = None) -> None:
        """Process audio data. Override this method to implement audio processing."""
        pass

    def should_process(self) -> bool:
        """Check if it's time to process based on the interval."""
        current_time = asyncio.get_event_loop().time()

        if current_time - self.last_process_time >= self.interval:
            self.last_process_time = current_time
            return True
        return False


class AudioLogger(AudioVideoProcessor):

    def __init__(self, interval: int = 2):
        super().__init__(interval, receive_audio=True, receive_video=False)
        self.audio_count = 0
        self.last_audio_time = 0

    async def process_audio(self, audio_data: bytes, user_id: str, metadata: dict = None) -> None:
        """Log audio data information."""
        current_time = asyncio.get_event_loop().time()

        if self.should_process():
            self.audio_count += 1

            logging.info(
                f"🎵 Audio #{self.audio_count} from {user_id}: "
                f"{len(audio_data)} bytes"
            )

class ImageCapture(AudioVideoProcessor):
    """Handles video frame capture and storage at regular intervals."""

    def __init__(self, output_dir: str = "captured_frames", interval: int = 3, *args, **kwargs):
        super().__init__(interval, receive_audio=False, receive_video=True, *args, **kwargs)
        self.output_dir = Path(output_dir)
        self.frame_count = 0

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        logging.info(f"📁 Saving captured frames to: {self.output_dir.absolute()}")

    async def process_image(self, image: Image.Image, user_id: str, metadata: dict = None):
        timestamp = int(asyncio.get_event_loop().time())
        filename = f"frame_{user_id}_{timestamp}_{self.frame_count:04d}.jpg"
        filepath = self.output_dir / filename

        # Save the frame as JPG
        image.save(filepath, "JPEG", quality=90)

        self.frame_count += 1
        logging.info(
            f"📸 Captured frame {self.frame_count}: {filename} ({frame.width}x{frame.height})"
        )

        return str(filepath)