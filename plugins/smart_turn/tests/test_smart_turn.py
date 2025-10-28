import asyncio
import os
import tempfile

import pytest

from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.turn_detection import TurnStartedEvent, TurnEndedEvent
from vision_agents.plugins.smart_turn.smart_turn_detection import (
    SileroVAD,
    SmartTurnDetection,
    ensure_model,
    SILERO_ONNX_URL,
    SILERO_ONNX_FILENAME,
)
import logging

logger = logging.getLogger(__name__)


class TestTurn2:
    @pytest.fixture
    async def td(self):
        td = SmartTurnDetection()
        await td.start()
        yield td

    async def test_silero_predict(self, mia_audio_16khz):
        path = os.path.join(tempfile.gettempdir(), SILERO_ONNX_FILENAME)
        await ensure_model(path, SILERO_ONNX_URL)
        vad = SileroVAD(path)

        for pcm_chunk in mia_audio_16khz.chunks(chunk_size=1024):
            result = vad.predict_speech(
                pcm_chunk.resample(target_sample_rate=16000).to_float32().samples
            )
            print(result)

    async def test_turn_detection_chunks(self, td, mia_audio_16khz):
        participant = Participant(user_id="mia", original={})
        conversation = InMemoryConversation(instructions="be nice", messages=[])

        event_order = []

        # Subscribe to events
        @td.events.subscribe
        async def on_start(event: TurnStartedEvent):
            logger.info(f"Smart turn turn started on {event.session_id}")
            event_order.append("start")

        @td.events.subscribe
        async def on_stop(event: TurnEndedEvent):
            logger.info(f"Smart turn turn ended on {event.session_id}")
            event_order.append("stop")

        for pcm in mia_audio_16khz.chunks(chunk_size=1024):
            await td.process_audio(pcm, participant, conversation)
            # simulate how td gets data periodically
            await asyncio.sleep(512.0 / 16000.0)

        await asyncio.sleep(4)
        assert event_order == ["start", "stop"]

    async def test_turn_detection(self, td, mia_audio_16khz):
        participant = Participant(user_id="mia", original={})
        conversation = InMemoryConversation(instructions="be nice", messages=[])
        event_order = []

        # Subscribe to events
        @td.events.subscribe
        async def on_start(event: TurnStartedEvent):
            logger.info(f"Smart turn turn started on {event.session_id}")
            event_order.append("start")

        @td.events.subscribe
        async def on_stop(event: TurnEndedEvent):
            logger.info(f"Smart turn turn ended on {event.session_id}")
            event_order.append("stop")

        await td.process_audio(mia_audio_16khz, participant, conversation)
        await asyncio.sleep(0.001)

        await asyncio.sleep(5)

        # Verify that turn detection is working - we should get at least some turn events
        # With continuous processing, we may get multiple start/stop cycles
        assert event_order == ["start", "stop"]
