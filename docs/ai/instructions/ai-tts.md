## TTS 

Here's a minimal example for building a new TTS plugin

```python


class MyTTS(tts.TTS):
    def __init__(
        self,
        voice_id: str = "VR6AewLTigWG4xSOukaG",  # Default ElevenLabs voice
        model_id: str = "eleven_multilingual_v2",
        client: Optional[MyClient] = None,
    ):
        # it should be possible to pass the client (makes it easier for users to customize things)
        # settings that are common to change, like voice id or model id should be configurable as well
        super().__init__()
        self.voice_id = voice_id
        self.client = client if client is not None else MyClient(api_key=api_key)

    async def stream_audio(self, text: str, *_, **__) -> AsyncIterator[bytes]:

        audio_stream = self.client.text_to_speech.stream(
            text=text,
            voice_id=self.voice_id,
            output_format=self.output_format,
            model_id=self.model_id,
            request_options={"chunk_size": 64000},
        )

        return audio_stream

```

TODO: the stop part can be generic
TODO: Track handling can be improved

## Testing your TTS

TOOD: no good test suite yet
