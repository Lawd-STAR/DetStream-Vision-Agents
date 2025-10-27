import asyncio
import os

import numpy as np
import pytest
from getstream.video.rtc.track_util import PcmData

from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.turn_detection import TurnStartedEvent, TurnEndedEvent
from vision_agents.plugins.smart_turn.turn_detection_2 import (
    ensure_model,
    SMART_TURN_ONNX_PATH,
    SMART_TURN_ONNX_URL,
    SILERO_ONNX_PATH,
    SILERO_ONNX_URL,
    SileroVAD,
    SmartTurnDetection,
    MODEL_BASE_DIR,
)
import logging

logger = logging.getLogger(__name__)

class TestTurn2:

    @pytest.fixture
    async def td(self):
        td = SmartTurnDetection()
        try:
            await td.start()
            yield td
        finally:
            await td.stop()

    async def test_smart_turn_download(self):
        os.makedirs(MODEL_BASE_DIR, exist_ok=True)
        await ensure_model(SMART_TURN_ONNX_PATH, SMART_TURN_ONNX_URL)

    async def test_silero_download(self):
        os.makedirs(MODEL_BASE_DIR, exist_ok=True)
        await ensure_model(SILERO_ONNX_PATH, SILERO_ONNX_URL)

    def test_smart_turn_predict(self, td, mia_audio_16khz):
        result = td.predict_endpoint(mia_audio_16khz)
        print(result)

    def test_silero_predict(self, mia_audio_16khz):
        vad = SileroVAD(SILERO_ONNX_PATH)
        # TODO: chunk in 512
        chunk = mia_audio_16khz.samples[:512]
        iteration = 0
        for i in range(100):
            chunk = mia_audio_16khz.samples[i*512:(i+1)*512]
            int16 = np.frombuffer(chunk, dtype=np.int16)
            f32 = (int16.astype(np.float32)) / 32768.0
            # 16khz, float32 on 512 chunk size
            pcm_chunk = PcmData(samples=f32, format="f32", sample_rate=16000)
            result = vad.prob(pcm_chunk.samples)
            print(result)

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