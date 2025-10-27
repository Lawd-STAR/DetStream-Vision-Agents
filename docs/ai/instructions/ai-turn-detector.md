## Turn Detector

Here's a minimal example

```python

class MyTurnDetector(TurnDetector):
    async def process_audio(
        self,
        audio_data: PcmData,
        participant: Participant,
        conversation: Optional[Conversation],
    ) -> None:
    
        self._emit_start_turn_event(TurnStartedEvent(participant=participant))
        self._emit_end_turn_event(TurnEndedEvent(participant=participant, confidence=0.7))

    def start(self):
        super().start()
        # Any custom model loading/ or other heavy prep steps go here
        
    def stop(self):
        super().stop()
        # cleanup time. start and stop are optional

```

### Testing turn detection

An example test suite for turn detection can be found in `smart_turn/tests/test_smart_turn.py`