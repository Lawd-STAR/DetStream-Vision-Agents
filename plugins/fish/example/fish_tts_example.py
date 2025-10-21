import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from vision_agents.core import User
from vision_agents.core.agents import Agent
from vision_agents.plugins import fish, getstream, deepgram, smart_turn, gemini

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [call_id=%(call_id)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def start_agent() -> None:
    """
    Example demonstrating Fish Audio TTS integration with Vision Agents.
    
    This example creates an agent that uses:
    - Fish Audio for text-to-speech (TTS)
    - Deepgram for speech-to-text (STT)
    - GetStream for edge/real-time communication
    - Smart Turn for turn detection
    
    Requirements:
    - FISH_AUDIO_API_KEY environment variable
    - DEEPGRAM_API_KEY environment variable
    - STREAM_API_KEY and STREAM_API_SECRET environment variables
    """
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Friendly AI"),
        instructions="You are a helpful AI assistant. Be friendly and conversational.",
        tts=fish.TTS(),  # Uses Fish Audio for text-to-speech
        stt=deepgram.STT(),
        llm=gemini.LLM("gemini-2.0-flash"),
        turn_detection=smart_turn.TurnDetection(buffer_duration=2.0, confidence_threshold=0.5),
    )
    await agent.create_user()

    call = agent.edge.client.video.call("default", str(uuid4()))
    await agent.edge.open_demo(call)

    with await agent.join(call):
        await asyncio.sleep(5)
        # The agent will greet the user using Fish Audio TTS
        await agent.llm.simple_response(text="Hello! I'm using Fish Audio for text-to-speech. How can I help you today?")
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(start_agent())

