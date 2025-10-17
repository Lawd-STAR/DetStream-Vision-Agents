from dataclasses import dataclass, field
from vision_agents.core.events import PluginBaseEvent
from typing import Optional, Any


@dataclass
class BedrockStreamEvent(PluginBaseEvent):
    """Event emitted when Bedrock provides a stream event."""
    type: str = field(default='plugin.bedrock.stream', init=False)
    event_data: Optional[Any] = None

