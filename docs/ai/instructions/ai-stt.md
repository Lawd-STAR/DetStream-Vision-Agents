## STT

```python
from vision_agents.core import stt

class MySTT(stt.STT):
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[MyClient] = None,
    ):
        super().__init__(provider_name="my_stt")
        # be sure to allow the passing of the client object
        # if client is not passed, create one
        # pass the most common settings for the client in the init (like api key)


    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ):
        parts = self.client.stt(pcm_data, stream=True)
        full_text = ""
        for part in parts:
            # parts that aren't finished
            self._emit_partial_transcript_event(part, participant, metadata)
            full_text += part
            
        # the full text
        self._emit_transcript_event(full_text, participant, metadata)

```

## Testing the STT

A good example of testing the STT can be found in plugins/fish/tests/test_fish_stt.py