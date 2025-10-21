## STT

```python
from vision_agents.core import stt

class MySTT(stt.STT):
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        sample_rate: int = 48000,
        client: Optional[AsyncDeepgramClient] = None,
    ):
        super().__init__(sample_rate=sample_rate)


    async def _process_audio_impl(
        self, pcm_data: PcmData, user_metadata: Optional[Union[Dict[str, Any], Participant]] = None
    ) -> Optional[List[Tuple[bool, str, Dict[str, Any]]]]:
        pass

```