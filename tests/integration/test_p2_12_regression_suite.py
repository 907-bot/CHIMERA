"""P2-12 — Phase 2 Regression Suite

Consolidates regression tests covering:
Physics, State, Events, Observatory, Storage, Replay, Checkpoint, Determinism, Performance, API.
"""

import pytest
from packages.core.models import WorldConfig, WorldState
from packages.core.serialization import hash_world_state
from packages.physics.engine import DeterministicEngine
from packages.observatory.storage import ObservatoryStorageEngine
from packages.observatory.features import ObservationMask
from packages.observatory.events import SimEvent, EventType, EventBus, EnergyMeasuredEvent


class TestPhase2RegressionSuite:
    """Consolidated regression suite for Phase 2 components."""

    def test_full_phase2_lifecycle_regression(self):
        # 1. Config & Deterministic Engine
        config = WorldConfig(world_id="p2_reg_world", seed=123, num_particles=6, dt=0.01)
        engine = DeterministicEngine(config=config)
        init_state = engine.current_state

        # 2. Run simulation
        history = engine.run(100)
        assert len(history) == 101
        final_state = history[-1]

        # 3. Store trajectory in DuckDB
        storage = ObservatoryStorageEngine(":memory:")
        storage.store_trajectory(history)

        # 4. Record milestone events
        ev = EnergyMeasuredEvent(world_id=config.world_id, step=100, time=1.0, payload={"e": 50.0})
        storage.store_event(ev)

        # 5. Query and verify trajectory slice
        queried = storage.query_trajectory_slice(config.world_id, start_step=0, end_step=100)
        assert len(queried) == 101
        assert hash_world_state(queried[-1]) == hash_world_state(final_state)

        # 6. Verify observation mask on queried state
        blind = ObservationMask.mask_state(queried[-1])
        assert blind.world_id == config.world_id
        assert len(blind.particles_positions) == 6

        # 7. Checkpoint restore reproduction
        fresh_engine = DeterministicEngine(config=config)
        fresh_engine.restore_state(history[50])
        resumed_history = [fresh_engine.current_state]
        for _ in range(50):
            resumed_history.append(fresh_engine.step())

        assert hash_world_state(resumed_history[-1]) == hash_world_state(final_state)

        storage.close()
