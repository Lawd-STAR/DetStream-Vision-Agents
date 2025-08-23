import os
import pytest
from uuid import uuid4

from dotenv import load_dotenv
from getstream import Stream
from getstream.models import UserRequest

from agents.conversation import Conversation


@pytest.mark.integration
class TestConversationIntegration:
    """Integration test suite for the Conversation class using real Stream API."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        load_dotenv()
        
        # Check if required environment variables are set
        api_key = os.getenv('STREAM_API_KEY')
        api_secret = os.getenv('STREAM_API_SECRET')
        
        if not api_key or not api_secret:
            pytest.skip("STREAM_API_KEY and STREAM_API_SECRET must be set for integration tests")
        
        # Initialize Stream client
        cls.client = Stream.from_env()
        
        # Create test users
        cls.agent_user_id = f"test-agent-{uuid4()}"
        cls.human_user_id = f"test-human-{uuid4()}"
        
        # Upsert users
        cls.client.upsert_users(
            UserRequest(id=cls.agent_user_id, name="Test Agent"),
            UserRequest(id=cls.human_user_id, name="Test Human")
        )

    @classmethod
    def teardown_class(cls):
        """Clean up test resources."""
        if hasattr(cls, 'client'):
            # Clean up users (optional, as test users are usually ephemeral)
            try:
                cls.client.delete_users(cls.agent_user_id, cls.human_user_id)
            except Exception:
                pass  # Ignore cleanup errors

    @pytest.fixture
    def test_channel_id(self):
        """Create a unique test channel ID."""
        return f"test-channel-{uuid4()}"

    @pytest.fixture
    def conversation(self, test_channel_id):
        """Create a Conversation instance with real Stream client and channel."""
        # Create channel
        channel_response = self.client.chat.get_or_create_channel(
            "messaging", 
            test_channel_id,
            data={"created_by_id": self.agent_user_id}
        ).data.channel
        
        # Create conversation
        conv = Conversation([], channel_response, self.client.chat)
        
        yield conv
        
        # Cleanup: Delete channel after test
        try:
            self.client.chat.delete_channel("messaging", test_channel_id)
        except Exception:
            pass  # Ignore cleanup errors

    def test_conversation_initialization(self, conversation):
        """Test that conversation initializes correctly with real Stream objects."""
        assert conversation.messages == []
        assert conversation.last_message is None
        assert conversation.channel is not None
        assert conversation.chat_client is not None
        assert conversation.channel.type == "messaging"

    def test_add_message_real_api(self, conversation):
        """Test adding a message using the real Stream API."""
        initial_count = len(conversation.messages)
        test_text = "Hello from integration test!"
        
        # Add message
        conversation.add_message(test_text, self.human_user_id)
        
        # Verify message was added locally
        assert len(conversation.messages) == initial_count + 1
        assert conversation.messages[-1].text == test_text
        assert conversation.messages[-1].user_id == self.human_user_id
        
        # For integration test, we verify that the _send_message method was called
        # without errors (which means the message was sent to Stream successfully)
        # The fact that we got here without an exception means the API call worked

    def test_update_last_message_create_new_real_api(self, conversation):
        """Test creating a new streaming message using real API."""
        test_text = "Streaming message test"
        test_user_id = self.agent_user_id
        
        # Create new streaming message
        conversation.update_last_message(test_text, test_user_id)
        
        # Verify last_message was set
        assert conversation.last_message is not None
        assert conversation.last_message.text == test_text
        assert conversation.last_message.user.id == test_user_id
        
        # Integration test: verify the message was successfully sent to Stream
        # (no exception means the API call worked)

    def test_update_last_message_append_real_api(self, conversation):
        """Test appending to existing streaming message using real API."""
        initial_text = "Initial streaming text"
        append_text = " - appended content"
        
        # Create initial message
        conversation.update_last_message(initial_text, self.agent_user_id)
        initial_message_id = conversation.last_message.id
        
        # Append to existing message
        conversation.update_last_message(append_text)
        
        # Verify text was appended locally
        expected_text = initial_text + append_text
        assert conversation.last_message.text == expected_text
        assert conversation.last_message.id == initial_message_id
        
        # Integration test: verify the append operation completed successfully
        # (no exception means the API call worked)

    def test_finish_last_message_real_api(self, conversation):
        """Test finishing a streaming message using real API."""
        test_text = "Message to be finished"
        
        # Create streaming message
        conversation.update_last_message(test_text, self.agent_user_id)
        message_id = conversation.last_message.id
        
        # Finish the message
        conversation.finish_last_message()
        
        # Verify last_message was cleared locally
        assert conversation.last_message is None
        
        # Integration test: verify the finish operation completed successfully
        # (no exception means the API call worked)

    def test_partial_update_message_with_user_real_api(self, conversation):
        """Test partial update with user object using real API."""
        # Create a mock user object (simulating what would come from STT)
        class MockUser:
            def __init__(self, user_id):
                self.user_id = user_id
        
        mock_user = MockUser(self.human_user_id)
        test_text = "Partial transcript from user"
        
        # Call partial update
        conversation.partial_update_message(test_text, mock_user)
        
        # Verify message was created/updated
        assert conversation.last_message is not None
        assert conversation.last_message.text == test_text
        assert conversation.last_message.user.id == self.human_user_id

    def test_multiple_messages_workflow_real_api(self, conversation):
        """Test a complete workflow with multiple messages using real API."""
        # Add a regular message
        conversation.add_message("User said: Hello!", self.human_user_id)
        
        # Start streaming response
        conversation.update_last_message("Agent is responding", self.agent_user_id)
        
        # Append to streaming response
        conversation.update_last_message("... thinking...")
        conversation.update_last_message("... here's my answer!")
        
        # Finish streaming response
        conversation.finish_last_message()
        
        # Verify local state
        assert len(conversation.messages) == 1  # Only the added message, streaming doesn't go to messages list
        assert conversation.last_message is None
        
        # Integration test: verify all operations completed successfully
        # (no exceptions means all API calls worked)

    def test_empty_channel_state(self, conversation):
        """Test conversation with empty channel (no existing messages)."""
        # Verify conversation starts with empty messages
        assert len(conversation.messages) == 0
        assert conversation.last_message is None
        
        # Add first message to the conversation
        conversation.add_message("First message in channel", self.human_user_id)
        
        # Verify message was added locally
        assert len(conversation.messages) == 1
        assert conversation.messages[0].text == "First message in channel"
        assert conversation.messages[0].user_id == self.human_user_id
        
        # Integration test: verify the message was successfully sent to Stream
        # (no exception means the API call worked)


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_conversation_integration.py -v -m integration
    pytest.main([__file__, "-v", "-m", "integration"])
