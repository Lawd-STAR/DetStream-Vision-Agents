import logging
from typing import List, Dict

from getstream.models import MessageRequest
from getstream.chat.async_channel import Channel

from vision_agents.core.agents.conversation import (
    InMemoryConversation,
    Message,
    StreamingMessageHandler,
)

logger = logging.getLogger(__name__)


class StreamStreamingMessageHandler(StreamingMessageHandler):
    """Stream implementation of StreamingMessageHandler."""

    def __init__(
        self,
        message_id: str,
        user_id: str,
        role: str,
        conversation: "StreamConversation" = None,
    ):
        super().__init__(message_id, user_id, role)
        self.conversation = conversation
        self.stream_message_id: str = None

    async def _on_content_created(self):
        """Create the message in Stream using send_message."""
        if self.conversation and self.conversation.channel:
            try:
                # Send the message to Stream
                request = MessageRequest(text=self.content, user_id=self.user_id)
                response = await self.conversation.channel.send_message(request)

                # Store the mapping between internal ID and Stream message ID
                self.stream_message_id = response.data.message.id
                self.conversation.internal_ids_to_stream_ids[self.message_id] = (
                    self.stream_message_id
                )

                logger.debug(
                    f"Created Stream message {self.stream_message_id} for internal message {self.message_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to create Stream message for {self.message_id}: {e}"
                )
                raise

        # Also update the message in the conversation's messages list
        if self.conversation:
            for message in self.conversation.messages:
                if message.id == self.message_id:
                    message.content = self.content
                    break

    async def _on_content_changed(self):
        """Update the message in Stream using ephemeral_message_update."""
        if self.conversation and self.conversation.channel and self.stream_message_id:
            try:
                await self.conversation.channel.client.ephemeral_message_update(
                    self.stream_message_id,
                    user_id=self.user_id,
                    set={"text": self.content, "generating": True},
                )
                logger.debug(
                    f"Updated Stream message {self.stream_message_id} with content: {self.content[:50]}..."
                )
            except Exception as e:
                logger.error(
                    f"Failed to update Stream message {self.stream_message_id}: {e}"
                )
                raise

        # Also update the message in the conversation's messages list
        if self.conversation:
            for message in self.conversation.messages:
                if message.id == self.message_id:
                    message.content = self.content
                    break

    async def _on_finalized(self):
        """Finalize the message in Stream using update_message_partial."""
        if self.conversation and self.conversation.channel and self.stream_message_id:
            try:
                await self.conversation.channel.client.update_message_partial(
                    self.stream_message_id,
                    user_id=self.user_id,
                    set={"text": self.content, "generating": False},
                )
                logger.debug(f"Finalized Stream message {self.stream_message_id}")
            except Exception as e:
                logger.error(
                    f"Failed to finalize Stream message {self.stream_message_id}: {e}"
                )
                raise

        # Remove the handler from the conversation when finalized
        if self.conversation:
            self.conversation._remove_handler(self.message_id)


class StreamConversation(InMemoryConversation):
    """
    Persists the message history to a stream channel & messages
    """

    messages: List[Message]

    # maps internal ids to stream message ids
    internal_ids_to_stream_ids: Dict[str, str]

    channel: Channel

    def __init__(self, instructions: str, messages: List[Message], channel: Channel):
        super().__init__(instructions, messages)
        self.messages = messages
        self.channel = channel
        self.internal_ids_to_stream_ids = {}

    def _create_handler(
        self, message_id: str, user_id: str, role: str
    ) -> StreamStreamingMessageHandler:
        """Create a new Stream streaming message handler."""
        # Create the message first (empty content initially, will be updated by handler)
        message = Message(content="", role=role, user_id=user_id, id=message_id)
        self.messages.append(message)

        # Create and return the handler
        return StreamStreamingMessageHandler(
            message_id=message_id, user_id=user_id, role=role, conversation=self
        )

    async def add_message(self, message: Message, completed: bool = True):
        """Add a message to the Stream conversation.

        Args:
            message: The Message object to add
            completed: If True, mark the message as completed using update_message_partial.
                      If False, mark as still generating using ephemeral_message_update.

        Returns:
            None
        """
        self.messages.append(message)

        # Send the message to Stream
        request = MessageRequest(text=message.content, user_id=message.user_id)
        response = await self.channel.send_message(request)

        # Store the mapping between internal ID and Stream message ID
        stream_id = response.data.message.id
        self.internal_ids_to_stream_ids[message.id] = stream_id

        # Update the message with completion status
        if completed:
            await self.channel.client.update_message_partial(
                stream_id,
                user_id=message.user_id,
                set={"text": message.content, "generating": False},
            )
        else:
            await self.channel.client.ephemeral_message_update(
                stream_id,
                user_id=message.user_id,
                set={"text": message.content, "generating": True},
            )

    async def update_message(
        self,
        message_id: str,
        input_text: str,
        user_id: str,
        replace_content: bool,
        completed: bool,
    ):
        """Update a message in the Stream conversation.

        This method updates both the local message content and syncs with the Stream API.
        If the message doesn't exist, it creates a new one.

        Args:
            message_id: The ID of the message to update
            input_text: The text content to set or append
            user_id: The ID of the user who owns the message
            replace_content: If True, replace the entire message content. If False, append to existing content.
            completed: If True, mark the message as completed using update_message_partial.
                      If False, mark as still generating using ephemeral_message_update.

        Returns:
            None
        """
        # First, update the local message using the superclass logic
        await super().update_message(
            message_id, input_text, user_id, replace_content, completed
        )

        # Get the updated message for Stream API sync
        message = self.lookup(message_id)
        if message is None:
            # This shouldn't happen after super().update_message, but handle gracefully
            logger.warning(f"message {message_id} not found after update")
            return None

        stream_id = self.internal_ids_to_stream_ids.get(message_id)
        if stream_id is None:
            logger.warning(
                f"stream_id for message {message_id} not found, skipping Stream API update"
            )
            return None

        # Update the message in Stream
        if completed:
            await self.channel.client.update_message_partial(
                stream_id,
                user_id=message.user_id,
                set={"text": message.content, "generating": False},
            )
        else:
            await self.channel.client.ephemeral_message_update(
                stream_id,
                user_id=message.user_id,
                set={"text": message.content, "generating": True},
            )
