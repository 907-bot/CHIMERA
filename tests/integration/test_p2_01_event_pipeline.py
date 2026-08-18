"""P2-01 — Event Pipeline Integration Coverage

Validates end-to-end event flow:
Simulation -> Physics state update -> Event creation -> Event bus -> Observatory -> Telemetry -> DuckDB -> Query -> Feature extraction
"""

import pytest
import numpy as np
from packages.core.models import WorldConfig, WorldState
from packages.physics.engine import DeterministicEngine
from packages.physics.energy import EnergyMetrics
from packages.observatory.events import (
    EventBus,
    EventType,
    SimEvent,
    ParticleCreatedEvent,
    CollisionEvent,
    EnergyMeasuredEvent,
    SnapshotRecordedEvent,
)
from packages.observatory.storage import ObservatoryStorageEngine
from packages.observatory.features import FeatureExtractor, ObservationMask


class TestEventPipelineIntegration:
    """End-to-end integration tests for the Event-Sourced Observatory pipeline."""

    def test_full_pipeline_flow(self):
        # 1. Setup EventBus and Storage
        bus = EventBus()
        storage = ObservatoryStorageEngine(":memory:")
        dispatched_events = []

        # Handlers to record to storage and capture dispatched events
        def event_handler(ev: SimEvent):
            dispatched_events.append(ev)
            storage.store_event(ev)

        for et in [
            EventType.PARTICLE_CREATED,
            EventType.COLLISION,
            EventType.ENERGY_MEASURED,
            EventType.SNAPSHOT_RECORDED,
        ]:
            bus.subscribe(et, event_handler)

        # 2. Simulation execution
        config = WorldConfig(world_id="pipeline_test_world", seed=42, num_particles=5, dt=0.01)
        engine = DeterministicEngine(config=config)

        # Publish initial particle creation events
        init_state = engine.current_state
        for p in init_state.particles:
            bus.publish(
                ParticleCreatedEvent(
                    world_id=init_state.world_id,
                    step=0,
                    time=0.0,
                    payload={"particle_id": p.id, "mass": p.mass, "radius": p.radius},
                )
            )

        # Run 50 steps
        history = [init_state]
        for step_idx in range(1, 51):
            state = engine.step()
            history.append(state)

            # Energy measurement milestone
            if step_idx % 10 == 0:
                energy_dict = EnergyMetrics.compute_all(state.particles)
                bus.publish(
                    EnergyMeasuredEvent(
                        world_id=state.world_id,
                        step=state.step,
                        time=state.time,
                        payload=energy_dict,
                    )
                )

            # Snapshot milestone
            if step_idx % 25 == 0:
                bus.publish(
                    SnapshotRecordedEvent(
                        world_id=state.world_id,
                        step=state.step,
                        time=state.time,
                        payload={"num_particles": len(state.particles)},
                    )
                )

        # 3. Batch store trajectory snapshots into DuckDB
        storage.store_trajectory(history)

        # 4. Acceptance Criteria Verification:
        # A. Events generated and arrived at observatory
        assert len(dispatched_events) == 5 + 5 + 2  # 5 created + 5 energy + 2 snapshot
        assert len(dispatched_events) == 12

        # B. Persisted events can be queried and match original
        queried_events = storage.query_events("pipeline_test_world")
        assert len(queried_events) == 12

        # C. No events silently lost and ordering is preserved
        steps_recorded = [ev.step for ev in queried_events]
        assert steps_recorded == sorted(steps_recorded)

        # D. Query trajectory slice matches simulation history exactly
        queried_states = storage.query_trajectory_slice("pipeline_test_world", start_step=0, end_step=50)
        assert len(queried_states) == 51

        for orig_s, q_s in zip(history, queried_states):
            assert orig_s.step == q_s.step
            assert abs(orig_s.time - q_s.time) < 1e-9
            assert len(orig_s.particles) == len(q_s.particles)
            for p_orig, p_q in zip(orig_s.particles, q_s.particles):
                assert p_orig.id == p_q.id
                assert abs(p_orig.position.x - p_q.position.x) < 1e-9
                assert abs(p_orig.position.y - p_q.position.y) < 1e-9
                assert abs(p_orig.velocity.x - p_q.velocity.x) < 1e-9
                assert abs(p_orig.velocity.y - p_q.velocity.y) < 1e-9

        # E. Feature extraction runs successfully on queried states
        entropy_values = [FeatureExtractor.spatial_entropy(s.particles) for s in queried_states]
        assert len(entropy_values) == 51
        assert all(isinstance(val, float) and val >= 0.0 for val in entropy_values)

        msd_series = FeatureExtractor.mean_squared_displacement(queried_states, particle_id=1)
        assert len(msd_series) == 51
        assert msd_series[0][1] == 0.0  # Displacement at t=0 is 0

        # F. Observation masking produces valid sanitized BlindObservation
        blind_obs = ObservationMask.mask_state(queried_states[10])
        assert blind_obs.world_id == "pipeline_test_world"
        assert blind_obs.step == 10
        assert len(blind_obs.particles_positions) == 5
        assert len(blind_obs.particles_velocities) == 5

        storage.close()

    def test_query_events_with_filter(self):
        storage = ObservatoryStorageEngine(":memory:")
        for step in range(10):
            ev1 = EnergyMeasuredEvent(world_id="w_filter", step=step, time=step * 0.01, payload={"e": step})
            storage.store_event(ev1)
            if step % 2 == 0:
                ev2 = SnapshotRecordedEvent(world_id="w_filter", step=step, time=step * 0.01, payload={"s": step})
                storage.store_event(ev2)

        energy_events = storage.query_events("w_filter", event_type="ENERGY_MEASURED")
        assert len(energy_events) == 10

        snapshot_events = storage.query_events("w_filter", event_type="SNAPSHOT_RECORDED")
        assert len(snapshot_events) == 5

        range_events = storage.query_events("w_filter", start_step=3, end_step=6)
        assert all(3 <= ev.step <= 6 for ev in range_events)
        storage.close()
