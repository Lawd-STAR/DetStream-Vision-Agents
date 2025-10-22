"""Tests for OpenRouter LLM plugin."""

import os

import pytest
from dotenv import load_dotenv

from vision_agents.core.agents.conversation import Message
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.plugins.openrouter import LLM

load_dotenv()


class TestOpenRouterLLM:
    """Test suite for OpenRouter LLM class."""

    def assert_response_successful(self, response):
        """Utility method to verify a response is successful.

        A successful response has:
        - response.text is set (not None and not empty)
        - response.exception is None

        Args:
            response: LLMResponseEvent to check
        """
        assert response.text is not None, "Response text should not be None"
        assert len(response.text) > 0, "Response text should not be empty"
        assert not hasattr(response, "exception") or response.exception is None, (
            f"Response should not have an exception, got: {getattr(response, 'exception', None)}"
        )

    def test_openrouter_llm_init(self):
        """Test that OpenRouter LLM can be initialized."""
        llm = LLM(api_key="test-key")
        assert llm is not None

    def test_openrouter_llm_base_url(self):
        """Test that OpenRouter LLM uses correct base URL."""
        llm = LLM(api_key="test-key")
        assert "openrouter.ai" in str(llm.client.base_url)

    def test_openrouter_llm_custom_model(self):
        """Test that OpenRouter LLM can use custom model."""
        llm = LLM(api_key="test-key", model="anthropic/claude-3-opus")
        assert llm.model == "anthropic/claude-3-opus"

    def test_message(self):
        """Test basic message normalization."""
        messages = LLM._normalize_message("say hi")
        assert isinstance(messages[0], Message)
        message = messages[0]
        assert message.original is not None
        assert message.content == "say hi"

    def test_advanced_message(self):
        """Test advanced message format with image."""
        img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/2023_06_08_Raccoon1.jpg/1599px-2023_06_08_Raccoon1.jpg"

        advanced = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what do you see in this image?"},
                    {"type": "input_image", "image_url": f"{img_url}"},
                ],
            }
        ]
        messages = LLM._normalize_message(advanced)
        assert messages[0].original is not None

    @pytest.fixture
    async def llm(self) -> LLM:
        """Fixture for OpenRouter LLM with z-ai/glm-4.6 model."""
        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY environment variable not set")
        
        llm = LLM(model="z-ai/glm-4.6")
        return llm

    @pytest.mark.integration
    async def test_simple(self, llm: LLM):
        """Test simple response generation."""
        response = await llm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )

        self.assert_response_successful(response)

    @pytest.mark.integration
    async def test_native_api(self, llm: LLM):
        """Test native OpenAI-compatible API."""
        response = await llm.create_response(
            input="say hi", instructions="You are a helpful assistant."
        )

        self.assert_response_successful(response)
        assert hasattr(response.original, "id")  # OpenAI-compatible response has id

    @pytest.mark.integration
    async def test_streaming(self, llm: LLM):
        """Test streaming response."""
        streamingWorks = False

        @llm.events.subscribe
        async def passed(event: LLMResponseChunkEvent):
            nonlocal streamingWorks
            streamingWorks = True

        response = await llm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )

        await llm.events.wait()

        self.assert_response_successful(response)
        assert streamingWorks, "Streaming should have generated chunk events"

    @pytest.mark.integration
    async def test_memory(self, llm: LLM):
        """Test conversation memory using simple_response."""
        await llm.simple_response(
            text="There are 2 dogs in the room",
        )
        response = await llm.simple_response(
            text="How many paws are there in the room?",
        )
        
        self.assert_response_successful(response)
        assert "8" in response.text or "eight" in response.text.lower(), (
            f"Expected '8' or 'eight' in response, got: {response.text}"
        )

    @pytest.mark.integration
    async def test_native_memory(self, llm: LLM):
        """Test conversation memory using native API."""
        await llm.create_response(
            input="There are 2 dogs in the room",
        )
        response = await llm.create_response(
            input="How many paws are there in the room?",
        )
        
        self.assert_response_successful(response)
        assert "8" in response.text or "eight" in response.text.lower(), (
            f"Expected '8' or 'eight' in response, got: {response.text}"
        )

    @pytest.mark.integration
    async def test_instruction_following(self):
        """Test that the LLM follows system instructions."""
        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY environment variable not set")
            
        llm = LLM(model="z-ai/glm-4.6")
        llm._set_instructions("Only reply in 2 letter country shortcuts")

        response = await llm.simple_response(
            text="Which country is rainy, protected from water with dikes and below sea level?",
        )

        self.assert_response_successful(response)
        assert "nl" in response.text.lower(), (
            f"Expected 'NL' in response, got: {response.text}"
        )

    @pytest.mark.integration
    async def test_events(self, llm: LLM):
        """Test that LLM events are properly emitted during streaming responses."""
        chunk_events = []
        complete_events = []

        @llm.events.subscribe
        async def handle_chunk_event(event: LLMResponseChunkEvent):
            chunk_events.append(event)

        @llm.events.subscribe
        async def handle_complete_event(event: LLMResponseCompletedEvent):
            complete_events.append(event)

        # Make API call that should generate streaming events
        response = await llm.create_response(
            input="Create a small story about the weather in the Netherlands. Make it at least 2 paragraphs long.",
        )

        # Wait for all events to be processed
        await llm.events.wait()

        # Verify response was generated
        self.assert_response_successful(response)
        assert len(response.text) > 50, "Response should be substantial"

        # Verify chunk events were emitted
        assert len(chunk_events) > 0, (
            "Should have received chunk events during streaming"
        )

        # Verify completion event was emitted
        assert len(complete_events) > 0, "Should have received completion event"
        assert len(complete_events) == 1, "Should have exactly one completion event"

        # Verify chunk events have proper content and item_id
        total_delta_text = ""
        chunk_item_ids = set()
        for chunk_event in chunk_events:
            assert chunk_event.delta is not None, (
                "Chunk events should have delta content"
            )
            assert isinstance(chunk_event.delta, str), "Delta should be a string"
            if chunk_event.item_id is not None:
                chunk_item_ids.add(chunk_event.item_id)
            total_delta_text += chunk_event.delta

        # Verify completion event has proper content
        complete_event = complete_events[0]
        assert complete_event.text == response.text, (
            "Completion event text should match response text"
        )
        assert complete_event.original is not None, (
            "Completion event should have original response"
        )

        # Verify that chunk deltas reconstruct the final text (approximately)
        assert len(total_delta_text) > 0, "Should have accumulated delta text"
        assert len(total_delta_text) >= len(response.text) * 0.8, (
            "Delta text should be substantial portion of final text"
        )

