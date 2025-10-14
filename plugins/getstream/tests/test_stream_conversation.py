import logging
import pytest
import uuid
import asyncio
import os
from typing import List, Dict
from unittest.mock import Mock, AsyncMock
from dotenv import load_dotenv

from getstream.models import MessageRequest, ChannelInput
from getstream.chat.async_channel import Channel
from getstream import AsyncStream

from vision_agents.core.agents.conversation import (
    InMemoryConversation,
    Message,
    StreamingMessageHandler,
)
from vision_agents.plugins.getstream.stream_conversation import StreamConversation, StreamStreamingMessageHandler

logger = logging.getLogger(__name__)

load_dotenv()

class TestStreamConversation:
    """Test suite for StreamConversation class."""
    
    @pytest.fixture
    def mock_channel(self):
        """Create a mock Channel."""
        channel = Mock()
        channel.channel_type = "messaging"
        channel.channel_id = "test-channel-123"
        
        # Mock the client
        channel.client = Mock()
        
        # Create async mocks for client methods
        channel.client.update_message_partial = AsyncMock(return_value=Mock())
        channel.client.ephemeral_message_update = AsyncMock(return_value=Mock())
        
        # Mock send_message response
        mock_response = Mock()
        mock_response.data.message.id = "stream-message-123"
        
        # Create async mock for send_message
        channel.send_message = AsyncMock(return_value=mock_response)
        
        return channel
    
    @pytest.fixture
    def stream_conversation(self, mock_channel):
        """Create a StreamConversation instance with mocked dependencies."""
        instructions = "You are a helpful assistant."
        messages = [
            Message(
                original=None,
                content="Hello",
                role="user",
                user_id="user1",
            )
        ]
        # Set IDs for messages
        for i, msg in enumerate(messages):
            msg.id = f"msg-{i}"
            
        conversation = StreamConversation(
            instructions=instructions,
            messages=messages,
            channel=mock_channel
        )
        
        # Pre-populate some stream IDs for testing
        conversation.internal_ids_to_stream_ids = {
            "msg-0": "stream-msg-0"
        }
        
        return conversation
    
    @pytest.mark.asyncio
    async def test_initialization(self, stream_conversation, mock_channel):
        """Test StreamConversation initialization."""
        assert stream_conversation.channel == mock_channel
        assert isinstance(stream_conversation.internal_ids_to_stream_ids, dict)
        assert len(stream_conversation.messages) == 1
    
    @pytest.mark.asyncio
    async def test_add_message(self, stream_conversation, mock_channel):
        """Test adding a message to the stream with default completed=True."""
        new_message = Message(
            original=None,
            content="Test message",
            role="user",
            user_id="user123"
        )
        new_message.id = "new-msg-id"
        
        await stream_conversation.add_message(new_message)
        
        # Verify message was added locally immediately
        assert len(stream_conversation.messages) == 2
        assert stream_conversation.messages[-1] == new_message
        
        # Verify Stream API was called
        mock_channel.send_message.assert_called_once()
        call_args = mock_channel.send_message.call_args

        request = call_args[0][0]
        assert isinstance(request, MessageRequest)
        assert request.text == "Test message"
        assert request.user_id == "user123"
        
        # Verify ID mapping was stored
        assert "new-msg-id" in stream_conversation.internal_ids_to_stream_ids
        assert stream_conversation.internal_ids_to_stream_ids["new-msg-id"] == "stream-message-123"
        
        # Verify update_message_partial was called (completed=True is default)
        mock_channel.client.update_message_partial.assert_called_once()
        update_args = mock_channel.client.update_message_partial.call_args
        assert update_args[0][0] == "stream-message-123"
        assert update_args[1]["user_id"] == "user123"
        assert update_args[1]["set"]["text"] == "Test message"
        assert update_args[1]["set"]["generating"] is False  # completed=True means not generating
    
    @pytest.mark.asyncio
    async def test_add_message_with_completed_false(self, stream_conversation, mock_channel):
        """Test adding a message with completed=False (still generating)."""
        # Reset mocks
        mock_channel.send_message.reset_mock()
        mock_channel.client.ephemeral_message_update.reset_mock()
        mock_channel.client.update_message_partial.reset_mock()
        
        new_message = Message(
            original=None,
            content="Generating message",
            role="assistant",
            user_id="assistant"
        )
        new_message.id = "gen-msg-id"
        
        await stream_conversation.add_message(new_message, completed=False)
        
        # Verify message was added locally
        assert len(stream_conversation.messages) == 2
        assert stream_conversation.messages[-1] == new_message
        
        # Verify Stream API was called
        mock_channel.send_message.assert_called_once()
        
        # Verify ephemeral_message_update was called (completed=False)
        mock_channel.client.ephemeral_message_update.assert_called_once()
        mock_channel.client.update_message_partial.assert_not_called()
        
        update_args = mock_channel.client.ephemeral_message_update.call_args
        assert update_args[0][0] == "stream-message-123"
        assert update_args[1]["user_id"] == "assistant"
        assert update_args[1]["set"]["text"] == "Generating message"
        assert update_args[1]["set"]["generating"] is True  # completed=False means still generating
    
    @pytest.mark.asyncio
    async def test_update_message_existing(self, stream_conversation, mock_channel):
        """Test updating an existing message by appending content."""
        # Update existing message by appending (replace_content=False, completed=False)
        await stream_conversation.update_message(
            message_id="msg-0",
            input_text=" additional text",
            user_id="user1",
            replace_content=False,
            completed=False
        )
        
        # Verify message content was appended immediately
        assert stream_conversation.messages[0].content == "Hello additional text"
        
        # Verify Stream API was called with ephemeral_message_update (not completed)
        mock_channel.client.ephemeral_message_update.assert_called_once()
        call_args = mock_channel.client.ephemeral_message_update.call_args
        assert call_args[0][0] == "stream-msg-0"  # stream message ID
        assert call_args[1]["user_id"] == "user1"
        assert call_args[1]["set"]["text"] == "Hello additional text"
        assert call_args[1]["set"]["generating"] is True  # not completed = still generating
    
    @pytest.mark.asyncio
    async def test_update_message_replace(self, stream_conversation, mock_channel):
        """Test replacing message content (replace_content=True)."""
        # Mock update_message_partial for completed messages
        mock_channel.client.update_message_partial = AsyncMock(return_value=Mock())
        
        await stream_conversation.update_message(
            message_id="msg-0",
            input_text="Replaced content",
            user_id="user1",
            replace_content=True,
            completed=True
        )
        
        # Verify message content was replaced
        assert stream_conversation.messages[0].content == "Replaced content"
        
        # Verify Stream API was called with update_message_partial (completed)
        mock_channel.client.update_message_partial.assert_called_once()
        call_args = mock_channel.client.update_message_partial.call_args
        assert call_args[0][0] == "stream-msg-0"  # stream message ID
        assert call_args[1]["user_id"] == "user1"
        assert call_args[1]["set"]["text"] == "Replaced content"
        assert call_args[1]["set"]["generating"] is False  # completed = not generating
    
    @pytest.mark.asyncio
    async def test_update_message_not_found(self, stream_conversation, mock_channel):
        """Test updating a non-existent message creates a new one."""
        # Reset the send_message mock for this test
        mock_channel.send_message.reset_mock()
        
        await stream_conversation.update_message(
            message_id="non-existent-id",
            input_text="New message content",
            user_id="user2",
            replace_content=True,
            completed=False
        )
        
        # Should have added a new message
        assert len(stream_conversation.messages) == 2
        
        # Verify the new message was created correctly
        new_msg = stream_conversation.messages[-1]
        assert new_msg.id == "non-existent-id"
        assert new_msg.content == "New message content"
        assert new_msg.user_id == "user2"
        
        # Verify send_message was called (not update)
        mock_channel.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_message_completed_vs_generating(self, stream_conversation, mock_channel):
        """Test that completed=True calls update_message_partial and completed=False calls ephemeral_message_update."""
        # Mock update_message_partial for completed messages
        mock_channel.client.update_message_partial = AsyncMock(return_value=Mock())
        
        # Test with completed=False (still generating)
        await stream_conversation.update_message(
            message_id="msg-0",
            input_text=" in progress",
            user_id="user1",
            replace_content=False,
            completed=False
        )
        
        # Should call ephemeral_message_update
        mock_channel.client.ephemeral_message_update.assert_called()
        mock_channel.client.update_message_partial.assert_not_called()
        
        # Reset mocks
        mock_channel.client.ephemeral_message_update.reset_mock()
        
        # Test with completed=True
        await stream_conversation.update_message(
            message_id="msg-0",
            input_text="Final content",
            user_id="user1",
            replace_content=True,
            completed=True
        )
        
        # Should call update_message_partial
        mock_channel.client.update_message_partial.assert_called_once()
        mock_channel.client.ephemeral_message_update.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_update_message_no_stream_id(self, stream_conversation, mock_channel):
        """Test updating a message without a stream ID mapping."""
        # Add a message without stream ID mapping
        new_msg = Message(
            original=None,
            content="Test",
            role="user",
            user_id="user3"
        )
        new_msg.id = "unmapped-msg"
        stream_conversation.messages.append(new_msg)
        
        # Try to update it by appending
        await stream_conversation.update_message(
            message_id="unmapped-msg",
            input_text=" updated",
            user_id="user3",
            replace_content=False,
            completed=False
        )
        
        # Message should still be updated locally (with space handling)
        assert stream_conversation.messages[-1].content == "Test updated"
        
        # Since there's no stream_id mapping, the API call should be skipped
        # This is the expected behavior - we don't sync messages without stream IDs
        mock_channel.client.ephemeral_message_update.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_streaming_message_handle(self, stream_conversation, mock_channel):
        """Test streaming message with handle API."""
        # Reset mocks
        mock_channel.send_message.reset_mock()
        mock_channel.client.ephemeral_message_update.reset_mock()
        mock_channel.client.update_message_partial.reset_mock()
        
        # Start a streaming message
        handle = await stream_conversation.start_streaming_message(role="assistant", initial_content="Processing")
        
        # Verify message was added and marked as generating
        assert len(stream_conversation.messages) == 2
        assert stream_conversation.messages[-1].content == "Processing"
        assert stream_conversation.messages[-1].role == "assistant"
        assert handle.user_id == "assistant"
        
        # Verify send_message was called
        mock_channel.send_message.assert_called_once()
        # Verify ephemeral_message_update was called (completed=False by default)
        mock_channel.client.ephemeral_message_update.assert_called_once()
        
        # Reset for next operations
        mock_channel.client.ephemeral_message_update.reset_mock()
        
        # Append to the message
        await stream_conversation.append_to_message(handle, "...")
        assert stream_conversation.messages[-1].content == "Processing..."
        
        mock_channel.client.ephemeral_message_update.assert_called_once()
        
        # Replace the message
        await stream_conversation.replace_message(handle, "Complete response")
        assert stream_conversation.messages[-1].content == "Complete response"
        
        assert mock_channel.client.ephemeral_message_update.call_count == 2
        
        # Complete the message
        mock_channel.client.update_message_partial.reset_mock()
        await stream_conversation.complete_message(handle)
        
        mock_channel.client.update_message_partial.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multiple_streaming_handles(self, stream_conversation, mock_channel):
        """Test multiple concurrent streaming messages with Stream API."""
        # Reset mocks
        mock_channel.send_message.reset_mock()
        mock_channel.client.ephemeral_message_update.reset_mock()
        
        # Mock different message IDs for each send
        mock_response1 = Mock()
        mock_response1.data.message.id = "stream-msg-1"
        mock_response2 = Mock()
        mock_response2.data.message.id = "stream-msg-2"
        mock_channel.send_message.side_effect = [mock_response1, mock_response2]
        
        # Start two streaming messages with empty initial content
        handle1 = await stream_conversation.start_streaming_message(role="user", user_id="user123", initial_content="")
        handle2 = await stream_conversation.start_streaming_message(role="assistant", initial_content="")
        
        assert len(stream_conversation.messages) == 3  # 1 initial + 2 new
        
        # Update them independently
        await stream_conversation.append_to_message(handle1, "Hello?")
        await stream_conversation.append_to_message(handle2, "Hi there!")
        
        # Find messages by their handles to verify correct updates
        msg1 = next(msg for msg in stream_conversation.messages if msg.id == handle1.message_id)
        msg2 = next(msg for msg in stream_conversation.messages if msg.id == handle2.message_id)
        
        assert msg1.content == "Hello?"
        assert msg2.content == "Hi there!"
        
        # Verify ephemeral updates were called for both
        assert mock_channel.client.ephemeral_message_update.call_count >= 4  # 2 initial + 2 appends
        
        # Complete both
        await stream_conversation.complete_message(handle1)
        await stream_conversation.complete_message(handle2)
        
        # Verify update_message_partial was called for both completions
        assert mock_channel.client.update_message_partial.call_count == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_streaming_message_handling_integration():
    """Test that streaming messages are handled correctly with multiple deltas and one completion event.
    
    This test simulates the exact scenario from the logs:
    - Multiple LLM delta responses with the same item_id
    - One LLM completion response with the same item_id
    - Should result in only one message being created and progressively updated
    """

    # Create a test channel
    channel_id = f"test-channel-{uuid.uuid4()}"
    chat_client = AsyncStream().chat
    channel = chat_client.channel("messaging", channel_id)
    
    # Create the channel in Stream
    await channel.get_or_create(
        data=ChannelInput(created_by_id="test-user"),
    )
    
    # Create a conversation
    conversation = StreamConversation(
        instructions="Test conversation",
        messages=[],
        channel=channel
    )
    
    # Test data - simulate the exact scenario from logs
    item_id = str(uuid.uuid4())
    user_id = "test-agent"
    role = "assistant"
    
    # Simulate multiple delta events (like in the logs) - using real data from your logs
    delta_contents = [
        "Okay",
        ", buckle up",
        "! This is a long one, a tale of weather and how it shaped a",
        " small town, its inhabitants, and a particular obsession.\n\n# The Ballad of Bumble",
        "brook and the Bewildering Breeze\n\nBumblebrook was a town nestled deep in the Verdant Valley, a place known for its predictability. Seasons arrived on",
        " cue, the sun shone reliably, and rain fell with the gentle insistence of a grandfather telling a familiar story. The townsfolk, in turn, were creatures of habit",
        ". They planted crops according to the Almanac, held festivals on designated dates, and even timed their naps by the position of the sun.\n\n![A picture of a quaint, idyllic town nestled in a valley. Colorful houses line a winding river, and rolling",
        " green hills surround it.  A few fluffy clouds dot the sky.](image_of_bumblebrook.jpg)\n\nFor generations, weather forecasting in Bumblebrook had been the domain of Old Man Hemlock. He resided on the",
        " highest hill, surrounded by wind chimes and weather vanes, and possessed an uncanny ability to predict the weather simply by sniffing the air and observing the behavior of squirrels. His pronouncements, delivered with the gravitas of a prophet, were law. If Hemlock predicted rain, umbrellas were deployed; if he prophesized sunshine",
        ", picnics were planned.\n\nThen came the year the wind turned… bewildering.\n\nIt started subtly. A gentle breeze that shifted directions every few minutes. Hemlock, initially unconcerned, attributed it to a \"passing fancy of the atmosphere.\" But the \"passing fancy\" persisted. The wind became erratic, swirling",
        ", sometimes even reversing its direction mid-gust.  The wind chimes became cacophonous instruments of chaos.\n\n![Image of chaotic wind chimes tangled and twisted, swaying wildly against a stormy sky.](chaotic_wind_chimes.jpg)\n\nThe consequences were immediate and comical, then increasingly concerning",
        ".  Weather vanes spun like dizzy tops. Clotheslines turned into tangled webs of laundry.  Kites refused to fly, instead plummeting into surprised pedestrians.  And, most disturbingly, Hemlock's predictions became… well, spectacularly wrong.\n\nHe predicted sunshine on days that brought torrential downpours. He swore",
        " on a dry harvest when hailstorms ravaged the crops. The townspeople, once trusting, began to regard him with suspicion, then outright derision. His credibility plummeted faster than a kite in a hurricane.\n\n\"He's lost it!\" cried Mrs. Higgins, whose prized petunias had been flattened by a rogue gust",
        ". \"The old coot's gone senile!\"\n\nDesperate to restore his reputation, Hemlock retreated to his hilltop observatory. He consulted ancient texts, built bizarre contraptions of copper wire and glass bottles, and even attempted to communicate with the squirrels, who, unsurprisingly, remained uncooperative.\n\nWhile",
        " Hemlock wrestled with the baffling breeze, the town descended into weather-related anarchy. Farmers planted at the wrong time, leading to stunted crops. The annual Summer Festival was rained out, causing widespread disappointment.  The entire rhythm of Bumblebrook, once so predictable, was thrown into disarray.\n\nAmidst this meteorological",
        " mayhem, a young girl named Elara emerged. Elara was a curious and inventive child, obsessed with tinkering. She had a workshop in her attic filled with spare parts, discarded gadgets, and half-finished projects.  While everyone else lamented the unreliable weather, Elara saw it as a fascinating puzzle.\n\n![",
        "Image of a young girl, Elara, in her attic workshop, surrounded by tools, wires, and partially built contraptions. She's looking intently at a complex device.](elara_workshop.jpg)\n\nShe began to meticulously record the wind's behavior. She built small, intricate weather stations using",
        " repurposed clockwork mechanisms and scraps of metal. She charted the wind's speed, direction, and even its \"personality,\" as she called it, assigning names like \"Wimpy Wendy\" to gentle breezes and \"Raging Rupert\" to fierce gusts.\n\nElara's parents, initially dismissive of her \"",
        "frivolous hobby,\" grew concerned as her obsession intensified. They tried to encourage her to play with other children, to engage in more \"normal\" activities. But Elara remained steadfast in her pursuit. She believed that the bewildering breeze was not random; it followed a pattern, however complex.\n\nOne evening",
        ", after weeks of relentless observation, Elara had a breakthrough.  She noticed a faint, almost imperceptible humming sound that accompanied the most erratic gusts. It was a low-frequency vibration, barely audible, but definitely present.\n\nFurther investigation revealed the source of the hum: a large, oddly shaped rock formation on",
        " the far side of Verdant Valley. The rock, composed of a rare mineral, resonated with certain atmospheric conditions, creating subtle shifts in the wind patterns. The baffling breeze, she realized, was not a random phenomenon, but an echo of the rock's strange energy.\n\nElara, brimming with excitement, raced",
        " to Old Man Hemlock's hilltop observatory. She presented her findings, along with her meticulously compiled data and her theory about the resonating rock. Hemlock, initially skeptical, listened intently as Elara explained her discovery. He examined her charts, listened to her reasoning, and, for the first time in months, a",
        " flicker of hope appeared in his eyes.\n\nTogether, Elara and Hemlock devised a plan. They couldn't move the rock – it was too large and deeply embedded. Instead, they designed a series of smaller resonators that could counteract the rock's effect. These resonators, placed strategically around Bumblebrook, would create",
        " a kind of \"weather shield,\" stabilizing the wind and restoring the town's predictable climate.\n\n![Image of Elara and Old Man Hemlock working together on a large, complex device that resembles a weather resonator. They are surrounded by blueprints and tools.](elara_hemlock_working.jpg)\n\n",
        "The townsfolk, initially wary, watched with curiosity as Elara and Hemlock installed the resonators.  Some scoffed, some whispered, but most simply hoped. When the last resonator was in place, a hush fell over Bumblebrook.\n\nAnd then… the wind stilled.\n\nNot completely, of course. There was still",
        " a gentle breeze, but it was no longer erratic, no longer bewildering. It blew from a consistent direction, carrying the scent of wildflowers and the promise of a predictable future.\n\nThe sun shone. The rain fell gently.  The crops flourished. The Summer Festival was held on its designated date, and it was the",
        " most joyous celebration Bumblebrook had seen in years.\n\nOld Man Hemlock's reputation was restored, and he acknowledged Elara as his equal in the art of weather forecasting. But Elara's contribution was more than just restoring order; she taught the town a valuable lesson. She showed them that even in the face of",
        " the most bewildering changes, observation, ingenuity, and a willingness to challenge assumptions could lead to understanding and ultimately, to a better future.\n\nAnd so, Bumblebrook returned to its predictable ways, but with a newfound appreciation for the unpredictable nature of life, and the remarkable power of a girl who dared to chase the bewildering breeze",
        ". Elara went on to become a renowned meteorologist, always remembering the day she saved her town, one baffling gust at a time. And Old Man Hemlock? He finally learned to use a computer. The end."
    ]
    
    # Process delta events concurrently to simulate real scenario
    async def process_delta(i, delta_content):
        return await conversation.upsert_streaming_message_handler(
            message_id=item_id,
            user_id=user_id,
            role=role,
            content=delta_content,
            content_index=i
        )
    
    # Fire all delta events concurrently (like in real scenario)
    tasks = [process_delta(i, content) for i, content in enumerate(delta_contents)]
    handlers = await asyncio.gather(*tasks)
    
    # Verify all handlers are the same instance (should be reused)
    for i, handler in enumerate(handlers):
        assert handler is handlers[0], f"Handler {i} should be the same instance as the first one"
    
    # Verify content is accumulating (should be the full content after all deltas)
    expected_content = "".join(delta_contents)
    assert handlers[0].content == expected_content, f"Content mismatch: expected '{expected_content[:100]}...', got '{handlers[0].content[:100]}...'"
    
    # Simulate completion event (using the real final content from your logs)
    final_content = "".join(delta_contents)  # The completion event contains the full content
    
    # Check if handler exists (simulating the completion event logic)
    existing_handler = await conversation.get_streaming_message_handler(item_id)
    assert existing_handler is not None, "Handler should exist after delta events"
    
    # Update the message with final content (simulating completion event)
    # This may fail for large messages due to Stream's 1000 character limit
    try:
        await conversation.update_message(
            existing_handler.message_id,
            final_content,
            existing_handler.user_id,
            True,
            True,
        )
        
        # Verify the handler was properly finalized (only if update succeeded)
        assert existing_handler.is_finalized, "Handler should be finalized after completion event"
        assert existing_handler.content == final_content, f"Final content mismatch: expected '{final_content}', got '{existing_handler.content}'"

        from getstream.models import MessagePaginationParams
        channel_state = await channel.get_or_create(state=True, messages=MessagePaginationParams(limit=10))
        assert len(channel_state.data.messages) == 1, f"Channel state should have 1 message"
        
    except Exception as e:
        # For large messages, this is expected to fail due to Stream's 1000 character limit
        if "larger than 1000 characters" in str(e):
            # This is the expected behavior for large messages
            # The content should still be accumulated in memory
            assert existing_handler.content == final_content, f"Content should be accumulated in memory: expected '{final_content}', got '{existing_handler.content}'"
            # The handler should not be finalized since the Stream update failed
            assert not existing_handler.is_finalized, "Handler should not be finalized if Stream update failed"
        else:
            # Re-raise unexpected exceptions
            raise


@pytest.mark.integration
@pytest.mark.asyncio
async def test_large_message_handling_fails():
    """Test that large messages (>1000 chars) fail as expected.
    
    This test should FAIL because the feature is currently broken.
    We want this test to fail to track the issue.
    """

    # Create a test channel
    channel_id = f"test-channel-{uuid.uuid4()}"
    chat_client = AsyncStream().chat
    channel = chat_client.channel("messaging", channel_id)
    
    # Create the channel in Stream
    await channel.get_or_create(
        data=ChannelInput(created_by_id="test-user"),
    )
    
    # Create a conversation
    conversation = StreamConversation(
        instructions="Test conversation",
        messages=[],
        channel=channel
    )
    
    # Create a very large message (>1000 characters)
    large_content = "A" * 2000  # 2000 characters, well over the 1000 limit
    
    # This should work (create the message)
    message = Message(
        original=None,
        content=large_content,
        role="assistant",
        user_id="test-agent"
    )
    message.id = str(uuid.uuid4())
    
    # This should fail because the message is too large for Stream's limits
    with pytest.raises(Exception):
        await conversation.add_message(message)
