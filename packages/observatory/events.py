"""Event-Sourced Logging Data Models and Pub/Sub Event Bus"""

from enum import Enum
from typing import Dict, Any, List, Callable, Optional
import uuid
import time
from pydantic import BaseModel, ConfigDict, Field
from packages.core.models import Vector2D


class EventType(str, Enum):
    PARTICLE_CREATED = "PARTICLE_CREATED"
    COLLISION = "COLLISION"
    ENERGY_MEASURED = "ENERGY_MEASURED"
    SNAPSHOT_RECORDED = "SNAPSHOT_RECORDED"
    PHASE_TRANSITION = "PHASE_TRANSITION"


class SimEvent(BaseModel):
    """Base Immutable Event-Sourced Log Entry."""
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    world_id: str
    step: int
    time: float
    event_type: EventType
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class ParticleCreatedEvent(SimEvent):
    """Event emitted when a new particle is instantiated."""
    event_type: EventType = EventType.PARTICLE_CREATED


class CollisionEvent(SimEvent):
    """Event emitted when a particle collision or boundary bounce occurs."""
    event_type: EventType = EventType.COLLISION


class EnergyMeasuredEvent(SimEvent):
    """Event emitted when energy metrics are calculated."""
    event_type: EventType = EventType.ENERGY_MEASURED


class SnapshotRecordedEvent(SimEvent):
    """Event emitted when a trajectory snapshot frame is stored."""
    event_type: EventType = EventType.SNAPSHOT_RECORDED


class EventBus:
    """In-Memory Event Dispatcher and Subscriber Registry."""

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[SimEvent], None]]] = {}

    def subscribe(self, event_type: EventType, handler: Callable[[SimEvent], None]):
        """Register a callback handler for a specific EventType."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def publish(self, event: SimEvent):
        """Publish an event to all registered subscribers."""
        if event.event_type in self._subscribers:
            for handler in self._subscribers[event.event_type]:
                handler(event)
