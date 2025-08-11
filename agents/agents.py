"""
Agent implementation for Stream video call integration.

This module provides the Agent class that allows for easy integration of AI agents
into Stream video calls with support for tools, pre-processors, and various AI services.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List, Optional, Protocol
from uuid import uuid4

from getstream.video import rtc
from getstream.video.rtc import audio_track

# Import STT and TTS base classes from stream-py package
from getstream.plugins.common.stt import STT
from getstream.plugins.common.tts import TTS


class Tool(Protocol):
    """Protocol for agent tools."""

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool."""
        ...


class PreProcessor(Protocol):
    """Protocol for pre-processors."""

    def process(self, data: Any) -> Any:
        """Process input data."""
        ...


class Model(Protocol):
    """Protocol for AI models."""

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from the model."""
        ...


# STT and TTS are now imported directly from stream-py package


class TurnDetection(Protocol):
    """Protocol for turn detection services."""

    def detect_turn(self, audio_data: bytes) -> bool:
        """Detect if it's the agent's turn to speak."""
        ...


class Agent:
    """
    AI Agent that can join Stream video calls and interact with participants.

    Example usage:
        agent = Agent(
            instructions="Roast my in-game performance in a funny but encouraging manner",
            tools=[dota_api("gameid")],
            pre_processors=[Roboflow()],
            model=openai_model,
            stt=speech_to_text,
            tts=text_to_speech,
            turn_detection=turn_detector
        )
        await agent.join(call)
    """

    def __init__(
        self,
        instructions: str,
        tools: Optional[List[Tool]] = None,
        pre_processors: Optional[List[PreProcessor]] = None,
        model: Optional[Model] = None,
        stt: Optional[STT] = None,
        tts: Optional[TTS] = None,
        turn_detection: Optional[TurnDetection] = None,
        bot_id: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the Agent.

        Args:
            instructions: System instructions for the agent
            tools: List of tools the agent can use
            pre_processors: List of pre-processors for input data
            model: AI model for generating responses
            stt: Speech-to-Text service
            tts: Text-to-Speech service
            turn_detection: Turn detection service
            bot_id: Unique bot ID (auto-generated if not provided)
            name: Display name for the bot
        """
        self.instructions = instructions
        self.tools = tools or []
        self.pre_processors = pre_processors or []
        self.model = model
        self.stt = stt
        self.tts = tts
        self.turn_detection = turn_detection
        self.bot_id = bot_id or f"agent-{uuid4()}"
        self.name = name or "AI Agent"

        self._connection: Optional[rtc.RTCConnection] = None
        self._audio_track: Optional[audio_track.AudioStreamTrack] = None
        self._is_running = False

        self.logger = logging.getLogger(f"Agent[{self.bot_id}]")

    async def join(
        self, call, user_creation_callback: Optional[Callable] = None
    ) -> None:
        """
        Join a Stream video call.

        Args:
            call: Stream video call object
            user_creation_callback: Optional callback to create the bot user
        """
        if self._is_running:
            raise RuntimeError("Agent is already running")

        self.logger.info(f"🤖 Agent joining call: {call.id}")

        # Create bot user if callback provided
        if user_creation_callback:
            user_creation_callback(self.bot_id, self.name)

        # Set up audio track if TTS is available
        if self.tts:
            self._audio_track = audio_track.AudioStreamTrack(framerate=16000)
            self.tts.set_output_track(self._audio_track)

        try:
            async with await rtc.join(call, self.bot_id) as connection:
                self._connection = connection
                self._is_running = True

                self.logger.info(f"🤖 Agent joined call: {call.id}")

                # Set up audio track if available
                if self._audio_track:
                    await connection.add_tracks(audio=self._audio_track)
                    self.logger.info("🤖 Agent ready to speak")

                # Set up event handlers
                await self._setup_event_handlers()

                # Send initial greeting if TTS is available
                if self.tts and self.instructions:
                    await self._send_initial_greeting()

                self.logger.info("🎧 Agent is active - press Ctrl+C to stop")
                await connection.wait()

        except asyncio.CancelledError:
            self.logger.info("Stopping agent...")
        except Exception as e:
            # Handle cleanup errors gracefully
            if "NoneType" in str(e) and "await" in str(e):
                self.logger.warning(
                    "Cleanup error (likely WebSocket already closed) - ignoring"
                )
            else:
                self.logger.error(f"Error during agent operation: {e}")
                raise
        finally:
            self._is_running = False
            self._connection = None

            # Clean up STT service
            if self.stt:
                try:
                    await self.stt.close()
                except Exception as e:
                    self.logger.warning(f"Error closing STT service: {e}")

    async def _setup_event_handlers(self) -> None:
        """Set up event handlers for the connection."""
        if not self._connection:
            return

        # Handle new participants
        async def on_track_published(event):
            try:
                if hasattr(event, "participant") and event.participant:
                    user_id = getattr(event.participant, "user_id", None)
                    if user_id and user_id != self.bot_id:
                        self.logger.info(f"👋 New participant joined: {user_id}")
                        await self._handle_new_participant(user_id)
            except Exception as e:
                self.logger.error(f"Error handling track published event: {e}")

        # Handle audio data for STT
        async def on_audio_received(event):
            try:
                if self.stt and hasattr(event, "audio_data"):
                    await self._handle_audio_input(event.audio_data)
            except Exception as e:
                self.logger.error(f"Error handling audio received event: {e}")

        # Safely set up event handlers
        try:
            if self._connection._ws_client:
                self._connection._ws_client.on_event(
                    "track_published", on_track_published
                )
                # Note: Audio handling would need to be implemented based on the actual Stream SDK
        except Exception as e:
            self.logger.error(f"Error setting up event handlers: {e}")

    async def _send_initial_greeting(self) -> None:
        """Send initial greeting to participants."""
        if not self.tts or not self._connection:
            return

        # Wait for publisher connection to be ready
        if self._connection.publisher_pc is not None:
            await self._connection.publisher_pc.wait_for_connected()

        # Check for existing participants
        existing_participants = [
            p
            for p in self._connection.participants_state._participant_by_prefix.values()
            if getattr(p, "user_id", None) != self.bot_id
        ]

        if existing_participants:
            greeting = await self._generate_greeting(len(existing_participants))
            try:
                await self.tts.send(greeting)
                self.logger.info("🤖 Sent initial greeting")
            except Exception as e:
                self.logger.error(f"Failed to send greeting: {e}")

    async def _handle_new_participant(self, user_id: str) -> None:
        """Handle a new participant joining the call."""
        if self.tts:
            greeting = await self._generate_participant_greeting(user_id)
            try:
                await self.tts.send(greeting)
                self.logger.info(f"🤖 Greeted new participant: {user_id}")
            except Exception as e:
                self.logger.error(f"Failed to greet participant {user_id}: {e}")

    async def _handle_audio_input(self, audio_data: bytes) -> None:
        """Handle incoming audio data."""
        if not self.stt:
            return

        try:
            # Check if it's our turn to respond
            if self.turn_detection and not self.turn_detection.detect_turn(audio_data):
                return

            # Set up event listener for transcription results
            if not hasattr(self, "_stt_setup"):
                self.stt.on("transcript", self._on_transcript)
                self.stt.on("partial_transcript", self._on_partial_transcript)
                self._stt_setup = True

            # Process audio using the PCM data format expected by stream-py STT
            # Note: This would need proper PCM data conversion in real implementation
            await self.stt.process_audio(audio_data)

        except Exception as e:
            self.logger.error(f"Error handling audio input: {e}")

    async def _on_transcript(self, text: str, user_metadata=None, metadata=None):
        """Handle final transcript from STT service."""
        if text and text.strip():
            self.logger.info(f"🎤 Heard: {text}")
            await self._process_transcription(text)

    async def _on_partial_transcript(
        self, text: str, user_metadata=None, metadata=None
    ):
        """Handle partial transcript from STT service."""
        # For now, we'll ignore partial transcripts to avoid responding to incomplete speech
        self.logger.debug(f"🎤 Partial: {text}")

    async def _process_transcription(self, text: str) -> None:
        """Process a complete transcription and generate response."""
        try:
            # Process with pre-processors
            processed_data = text
            for processor in self.pre_processors:
                processed_data = processor.process(processed_data)

            # Generate response using model
            if self.model:
                response = await self._generate_response(processed_data)

                # Send response via TTS
                if self.tts and response:
                    await self.tts.send(response)
                    self.logger.info(f"🤖 Responded: {response}")

        except Exception as e:
            self.logger.error(f"Error processing transcription: {e}")

    async def _generate_response(self, input_text: str) -> str:
        """Generate a response using the AI model."""
        if not self.model:
            return ""

        try:
            # Create context with instructions and available tools
            context = f"""
System: {self.instructions}

Available tools: {[str(tool) for tool in self.tools]}

User input: {input_text}

Respond appropriately based on your instructions.
"""

            response = await self.model.generate(context)
            return response

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error processing your request."

    async def _generate_greeting(self, participant_count: int) -> str:
        """Generate an initial greeting message."""
        if self.model:
            prompt = f"""
System: {self.instructions}

Generate a brief greeting for {participant_count} participant(s) who are already in the call.
Keep it friendly and contextual to your role.
"""
            try:
                return await self.model.generate(prompt)
            except Exception as e:
                self.logger.error(f"Error generating greeting: {e}")

        return f"Hello everyone! I'm {self.name} and I'm ready to help."

    async def _generate_participant_greeting(self, user_id: str) -> str:
        """Generate a greeting for a new participant."""
        if self.model:
            prompt = f"""
System: {self.instructions}

A new participant with ID '{user_id}' just joined the call.
Generate a brief, friendly greeting for them.
"""
            try:
                return await self.model.generate(prompt)
            except Exception as e:
                self.logger.error(f"Error generating participant greeting: {e}")

        return f"Welcome {user_id}! Great to have you here."

    def is_running(self) -> bool:
        """Check if the agent is currently running."""
        return self._is_running

    async def stop(self) -> None:
        """Stop the agent."""
        if self._connection:
            # This would need to be implemented based on the actual Stream SDK
            pass

        # Clean up STT service
        if self.stt:
            try:
                await self.stt.close()
            except Exception as e:
                self.logger.warning(f"Error closing STT service: {e}")

        self._is_running = False
