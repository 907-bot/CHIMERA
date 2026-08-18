"""P2-04 — DuckDB Data Integrity

Verifies DuckDB storage correctness, persistence across database file reopening,
aggregation queries, multi-experiment isolation, and large-dataset ingestion.
"""

import os
import pytest
import tempfile
from packages.core.models import WorldConfig, WorldState, Particle, Vector2D
from packages.physics.engine import DeterministicEngine
from packages.observatory.storage import ObservatoryStorageEngine
from packages.observatory.events import EnergyMeasuredEvent, SnapshotRecordedEvent


class TestDuckDBDataIntegrity:
    """Test suite for DuckDB data integrity and persistence."""

    def test_persistence_across_reopen(self, tmp_path):
        db_file = str(tmp_path / "telemetry_test.duckdb")

        # 1. Initialize and write data
        storage1 = ObservatoryStorageEngine(db_path=db_file)
        config = WorldConfig(world_id="persisted_world", seed=55, num_particles=6, dt=0.01)
        engine = DeterministicEngine(config=config)
        history = engine.run(50)
        storage1.store_trajectory(history)

        ev = EnergyMeasuredEvent(world_id="persisted_world", step=50, time=0.5, payload={"e": 100.0})
        storage1.store_event(ev)

        assert storage1.count_recorded_steps("persisted_world") == 51
        storage1.close()

        # 2. Re-open database file with fresh connection
        storage2 = ObservatoryStorageEngine(db_path=db_file)
        assert storage2.count_recorded_steps("persisted_world") == 51

        queried_states = storage2.query_trajectory_slice("persisted_world")
        assert len(queried_states) == 51
        assert queried_states[-1].step == 50
        assert len(queried_states[-1].particles) == 6

        queried_events = storage2.query_events("persisted_world")
        assert len(queried_events) == 1
        assert queried_events[0].payload == {"e": 100.0}

        storage2.close()

    def test_empty_dataset_query(self):
        storage = ObservatoryStorageEngine(":memory:")
        assert storage.query_trajectory_slice("nonexistent_world") == []
        assert storage.query_events("nonexistent_world") == []
        assert storage.count_recorded_steps("nonexistent_world") == 0
        storage.close()

    def test_multiple_experiments_isolation(self):
        storage = ObservatoryStorageEngine(":memory:")

        # World A
        cfg_a = WorldConfig(world_id="world_A", seed=1, num_particles=3)
        hist_a = DeterministicEngine(cfg_a).run(20)
        storage.store_trajectory(hist_a)

        # World B
        cfg_b = WorldConfig(world_id="world_B", seed=2, num_particles=4)
        hist_b = DeterministicEngine(cfg_b).run(30)
        storage.store_trajectory(hist_b)

        assert storage.count_recorded_steps("world_A") == 21
        assert storage.count_recorded_steps("world_B") == 31

        states_a = storage.query_trajectory_slice("world_A")
        states_b = storage.query_trajectory_slice("world_B")

        assert len(states_a) == 21
        assert len(states_b) == 31
        assert all(s.world_id == "world_A" for s in states_a)
        assert all(s.world_id == "world_B" for s in states_b)
        storage.close()

    def test_large_dataset_batch_ingestion(self):
        storage = ObservatoryStorageEngine(":memory:")
        cfg = WorldConfig(world_id="large_world", seed=99, num_particles=20)
        hist = DeterministicEngine(cfg).run(100)  # 101 steps * 20 particles = 2,020 rows
        storage.store_trajectory(hist)

        assert storage.count_recorded_steps("large_world") == 101
        states = storage.query_trajectory_slice("large_world", start_step=20, end_step=80)
        assert len(states) == 61
        assert all(len(s.particles) == 20 for s in states)
        storage.close()

    def test_single_state_insert(self):
        storage = ObservatoryStorageEngine(":memory:")
        cfg = WorldConfig(world_id="single_insert_world", seed=12, num_particles=3)
        state = DeterministicEngine(cfg).current_state
        storage.store_world_state(state)

        queried = storage.query_trajectory_slice("single_insert_world")
        assert len(queried) == 1
        assert queried[0].step == 0
        assert len(queried[0].particles) == 3
        storage.close()
