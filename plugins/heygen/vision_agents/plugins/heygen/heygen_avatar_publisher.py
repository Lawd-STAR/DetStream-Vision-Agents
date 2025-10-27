import asyncio
import logging
from typing import Optional, Any, Tuple

from vision_agents.core.processors.base_processor import (
    AudioVideoProcessor,
    VideoPublisherMixin,
)

from .heygen_rtc_manager import HeyGenRTCManager
from .heygen_video_track import HeyGenVideoTrack

logger = logging.getLogger(__name__)


class AvatarPublisher(AudioVideoProcessor, VideoPublisherMixin):
    """HeyGen avatar video publisher.
    
    Publishes video of a HeyGen avatar that lip-syncs to audio input.
    Can be used as a processor in the Vision Agents framework to add
    realistic avatar video to AI agents.
    
    Example:
        agent = Agent(
            edge=getstream.Edge(),
            agent_user=User(name="Avatar AI"),
            instructions="Be helpful and friendly",
            llm=gemini.LLM("gemini-2.0-flash"),
            tts=cartesia.TTS(),
            stt=deepgram.STT(),
            processors=[
                heygen.AvatarPublisher(
                    avatar_id="default",
                    quality="high"
                )
            ]
        )
    """

    def __init__(
        self,
        avatar_id: str = "default",
        quality: str = "high",
        resolution: Tuple[int, int] = (1920, 1080),
        api_key: Optional[str] = None,
        interval: int = 0,
        **kwargs,
    ):
        """Initialize the HeyGen avatar publisher.
        
        Args:
            avatar_id: HeyGen avatar ID to use for streaming.
            quality: Video quality ("low", "medium", "high").
            resolution: Output video resolution (width, height).
            api_key: HeyGen API key. Uses HEYGEN_API_KEY env var if not provided.
            interval: Processing interval (not used, kept for compatibility).
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(
            interval=interval,
            receive_audio=False,
            receive_video=False,
            **kwargs
        )
        
        self.avatar_id = avatar_id
        self.quality = quality
        self.resolution = resolution
        self.api_key = api_key
        
        # WebRTC manager for HeyGen connection
        self.rtc_manager = HeyGenRTCManager(
            avatar_id=avatar_id,
            quality=quality,
            api_key=api_key,
        )
        
        # Video track for publishing avatar frames
        self._video_track = HeyGenVideoTrack(
            width=resolution[0],
            height=resolution[1],
        )
        
        # Connection state
        self._connected = False
        self._connection_task: Optional[asyncio.Task] = None
        
        logger.info(
            f"🎭 HeyGen AvatarPublisher initialized "
            f"(avatar: {avatar_id}, quality: {quality}, resolution: {resolution})"
        )

    async def _connect_to_heygen(self) -> None:
        """Establish connection to HeyGen and start receiving video."""
        try:
            # Set up video callback before connecting
            self.rtc_manager.set_video_callback(self._on_video_track)
            
            # Connect to HeyGen
            await self.rtc_manager.connect()
            
            self._connected = True
            logger.info("✅ Connected to HeyGen, avatar streaming active")
        
        except Exception as e:
            logger.error(f"❌ Failed to connect to HeyGen: {e}")
            self._connected = False
            raise

    async def _on_video_track(self, track: Any) -> None:
        """Callback when video track is received from HeyGen.
        
        Args:
            track: Incoming video track from HeyGen's WebRTC connection.
        """
        logger.info("📹 Received video track from HeyGen, starting frame forwarding")
        await self._video_track.start_receiving(track)

    def publish_video_track(self):
        """Publish the HeyGen avatar video track.
        
        This method is called by the Agent to get the video track
        for publishing to the call.
        
        Returns:
            HeyGenVideoTrack instance for streaming avatar video.
        """
        # Start connection if not already connected
        if not self._connected and not self._connection_task:
            self._connection_task = asyncio.create_task(self._connect_to_heygen())
        
        logger.info("🎥 Publishing HeyGen avatar video track")
        return self._video_track

    def state(self) -> dict:
        """Get current state of the avatar publisher.
        
        Returns:
            Dictionary containing current state information.
        """
        return {
            "avatar_id": self.avatar_id,
            "quality": self.quality,
            "resolution": self.resolution,
            "connected": self._connected,
            "rtc_connected": self.rtc_manager.is_connected,
        }

    async def close(self) -> None:
        """Clean up resources and close connections."""
        logger.info("🔌 Closing HeyGen avatar publisher")
        
        # Stop video track
        if self._video_track:
            self._video_track.stop()
        
        # Close RTC connection
        if self.rtc_manager:
            await self.rtc_manager.close()
        
        # Cancel connection task if running
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
        
        self._connected = False
        logger.info("✅ HeyGen avatar publisher closed")

