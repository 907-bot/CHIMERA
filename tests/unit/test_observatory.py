"""Unit Tests for Observatory Event System, DuckDB Columnar Storage, and Feature Extraction"""

import pytest
import numpy as np
from packages.core.models import WorldConfig, WorldState, Particle, Vector2D, Boundary
from packages.physics.engine import DeterministicEngine
from packages.observatory.events import (
    EventBus,
    SimEvent,
    EventType,
    CollisionEvent,
    ParticleCreatedEvent,
)
from packages.observatory.storage import ObservatoryStorageEngine
from packages.observatory.features import FeatureExtractor, ObservationMask


def test_event_bus_pub_sub():
    bus = EventBus()
    received_events = []

    def handler(ev: SimEvent):
        received_events.append(ev)

    bus.subscribe(EventType.COLLISION, handler)

    ev1 = CollisionEvent(
        world_id="w1",
        step=10,
        time=0.1,
        payload={"particles": [1, 2]},
    )
    bus.publish(ev1)

    assert len(received_events) == 1
    assert received_events[0].world_id == "w1"
    assert received_events[0].event_type == EventType.COLLISION


def test_duckdb_storage_engine():
    storage = ObservatoryStorageEngine(":memory:")
    config = WorldConfig(world_id="test_duckdb", seed=42, num_particles=5)
    engine = DeterministicEngine(config=config)
    history = engine.run(20)

    # Store trajectory
    storage.store_trajectory(history)

    assert storage.count_recorded_steps("test_duckdb") == 21

    # Slice query [5, 15]
    slice_states = storage.query_trajectory_slice("test_duckdb", start_step=5, end_step=15)
    assert len(slice_states) == 11
    assert slice_states[0].step == 5
    assert slice_states[-1].step == 15

    storage.close()


def test_spatial_entropy():
    # 4 particles in distinct corners of 100x100 grid -> high entropy
    p1 = Particle(id=1, position=Vector2D(x=5.0, y=5.0))
    p2 = Particle(id=2, position=Vector2D(x=95.0, y=5.0))
    p3 = Particle(id=3, position=Vector2D(x=5.0, y=95.0))
    p4 = Particle(id=4, position=Vector2D(x=95.0, y=95.0))

    entropy = FeatureExtractor.spatial_entropy([p1, p2, p3, p4], grid_size=10)
    assert entropy > 0.0
    # 4 uniform cells occupied out of 100 -> S = - 4 * (0.25 * ln(0.25)) = ln(4) = ~1.38629
    assert entropy == pytest.approx(np.log(4.0), abs=1e-3)


def test_observation_mask_sanitization():
    config = WorldConfig(world_id="test_mask", gravity_constant=9.81, softening=0.123)
    engine = DeterministicEngine(config=config)
    state = engine.current_state

    blind_obs = ObservationMask.mask_state(state)

    assert blind_obs.world_id == "test_mask"
    assert len(blind_obs.particles_positions) == len(state.particles)
    # Hidden laws (gravity_constant, softening, force formulas) are stripped from BlindObservation
    assert not hasattr(blind_obs, "gravity_constant")
    assert not hasattr(blind_obs, "softening")
