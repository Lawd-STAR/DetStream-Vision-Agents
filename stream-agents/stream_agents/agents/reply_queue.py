



class ReplyQueue:
    """
    When a user interrupts the LLM, there are a few different behaviours that should be supported.
    1. Cancel/stop the audio playback, STT and LLM
    2. Pause and resume. Update context. Maybe reply the same
    3. Pause and refresh.

    Generating a reply, should write on this queue


    """

    def __init__(self, agent):
        self.agent = agent

    def pause(self):
        # TODO: some audio fade
        pass

    async def resume(self, text):
        # Some logic to either refresh (clear old) or simply resume
        response = await self.agent.llm.generate(text)
        self.agent.conversation.add_message(response, self.agent.agent_user.id)

        # TODO: streaming here to update messages

        await self.say_text(response)

    def _clear(self):
        pass

    async def say_text(self, text):
        # TODO: Stream and buffer
        await self.agent.tts.send(text)

    async def send_audio(self, pcm):
        # TODO: stream & buffer
        await self.agent._audio_track.send_audio(pcm)