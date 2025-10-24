import asyncio

import pytest

from plugins.smart_turn.vision_agents.plugins.smart_turn import TurnDetection
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.turn_detection import TurnStartedEvent, TurnEndedEvent

import logging

logger = logging.getLogger(__name__)

class TestSmartTurnTD:

    @pytest.fixture
    async def td(self):
        td = TurnDetection()
        try:
            td.start()
            yield td
        finally:
            td.stop()

    async def test_turn_detection(self, td, mia_audio_48khz_chunked):
        participant = Participant(user_id="mia", original={})
        conversation = InMemoryConversation(instructions="be nice", messages=[])

        # Subscribe to events
        @td.events.subscribe
        async def on_start(event: TurnStartedEvent):
            logger.info(f"Smart turn turn started on {event.session_id}")

        @td.events.subscribe
        async def on_stop(event: TurnEndedEvent):
            logger.info(f"Smart turn turn ended on {event.session_id}")

        # Process each 20ms audio chunk
        for chunk in mia_audio_48khz_chunked:
            await td.process_audio(chunk, participant, conversation)

        await asyncio.sleep(2)