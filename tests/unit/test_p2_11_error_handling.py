"""P2-11 — Error Handling

Verifies robust and deterministic error handling across EventBus, Storage,
Pydantic model validation, and serialization.
"""

import pytest
import pydantic
from packages.core.models import Vector2D, Particle, WorldConfig, WorldState
from packages.observatory.events import EventBus, EventType, SimEvent
from packages.observatory.storage import ObservatoryStorageEngine


class TestObservatoryErrorHandling:
    """Test suite for error boundary handling."""

    def test_invalid_vector_nan_validation(self):
        with pytest.raises(pydantic.ValidationError):
            Vector2D(x="not_a_number", y=10.0)

    def test_empty_trajectory_storage_safe(self):
        storage = ObservatoryStorageEngine(":memory:")
        # Storing empty trajectory should be a safe no-op
        storage.store_trajectory([])
        assert storage.count_recorded_steps("any") == 0
        storage.close()

    def test_operations_on_closed_storage_raise_cleanly(self):
        storage = ObservatoryStorageEngine(":memory:")
        storage.close()
        with pytest.raises(Exception):
            storage.query_trajectory_slice("world")

    def test_event_bus_unsubscribed_event_safe(self):
        bus = EventBus()
        ev = SimEvent(
            world_id="w",
            step=0,
            time=0.0,
            event_type=EventType.COLLISION,
            payload={},
        )
        # Publishing with no subscribers should not crash
        bus.publish(ev)
