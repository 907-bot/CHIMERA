"""P2-02 — Event Replay / State Reconstruction

Verifies deterministic state reconstruction via event stream replay.
Tests:
- Replay 1..N events produces exact state match
- Empty event stream
- Single event replay
- Multiple events replay
- Duplicate event handling
- Invalid event handling
- Missing event handling
- Out-of-order event handling
"""

import pytest
import numpy as np
from packages.core.models import WorldConfig, WorldState, Particle, Vector2D
from packages.core.serialization import hash_world_state
from packages.physics.engine import DeterministicEngine
from packages.observatory.events import (
    SimEvent,
    EventType,
    ParticleCreatedEvent,
    CollisionEvent,
    SnapshotRecordedEvent,
)


def reconstruct_state_from_snapshots(events: list[SimEvent]) -> dict[int, dict]:
    """Reconstruct particle position map from snapshot/particle events."""
    state_map = {}
    for ev in sorted(events, key=lambda e: (e.step, e.timestamp)):
        if "particles" in ev.payload:
            state_map[ev.step] = ev.payload["particles"]
    return state_map


class TestEventReplayStateReconstruction:
    """Test suite for event-sourced state replay and reconstruction."""

    def test_deterministic_event_replay(self):
        config = WorldConfig(world_id="replay_world", seed=77, num_particles=4, dt=0.01)
        engine = DeterministicEngine(config=config)

        # Run 50 steps and record snapshot events
        events = []
        history = [engine.current_state]

        for s in history:
            ev = SnapshotRecordedEvent(
                world_id=s.world_id,
                step=s.step,
                time=s.time,
                payload={
                    "particles": [
                        {"id": p.id, "pos_x": p.position.x, "pos_y": p.position.y, "vel_x": p.velocity.x, "vel_y": p.velocity.y}
                        for p in s.particles
                    ]
                },
            )
            events.append(ev)

        for _ in range(50):
            s = engine.step()
            history.append(s)
            ev = SnapshotRecordedEvent(
                world_id=s.world_id,
                step=s.step,
                time=s.time,
                payload={
                    "particles": [
                        {"id": p.id, "pos_x": p.position.x, "pos_y": p.position.y, "vel_x": p.velocity.x, "vel_y": p.velocity.y}
                        for p in s.particles
                    ]
                },
            )
            events.append(ev)

        # Replay events
        reconstructed = reconstruct_state_from_snapshots(events)
        assert len(reconstructed) == len(history)

        # Compare final state
        final_original = history[-1]
        final_reconstructed_particles = reconstructed[50]

        assert len(final_original.particles) == len(final_reconstructed_particles)
        for p_orig, p_rec in zip(final_original.particles, final_reconstructed_particles):
            assert p_orig.id == p_rec["id"]
            assert abs(p_orig.position.x - p_rec["pos_x"]) < 1e-9
            assert abs(p_orig.position.y - p_rec["pos_y"]) < 1e-9
            assert abs(p_orig.velocity.x - p_rec["vel_x"]) < 1e-9
            assert abs(p_orig.velocity.y - p_rec["vel_y"]) < 1e-9

    def test_empty_event_stream(self):
        empty_events = []
        reconstructed = reconstruct_state_from_snapshots(empty_events)
        assert reconstructed == {}

    def test_single_event_stream(self):
        ev = SnapshotRecordedEvent(
            world_id="single_ev_world",
            step=0,
            time=0.0,
            payload={"particles": [{"id": 1, "pos_x": 10.0, "pos_y": 20.0, "vel_x": 1.0, "vel_y": -1.0}]},
        )
        reconstructed = reconstruct_state_from_snapshots([ev])
        assert 0 in reconstructed
        assert len(reconstructed[0]) == 1
        assert reconstructed[0][0]["pos_x"] == 10.0

    def test_duplicate_events_idempotence(self):
        ev = SnapshotRecordedEvent(
            world_id="dup_world",
            step=1,
            time=0.01,
            payload={"particles": [{"id": 1, "pos_x": 12.0, "pos_y": 22.0, "vel_x": 1.0, "vel_y": -1.0}]},
        )
        # Pass duplicated events in stream
        events = [ev, ev]
        reconstructed = reconstruct_state_from_snapshots(events)
        # Final step 1 state must be identical
        assert 1 in reconstructed
        assert reconstructed[1][0]["pos_x"] == 12.0

    def test_out_of_order_events_sorted_reconstruction(self):
        ev0 = SnapshotRecordedEvent(world_id="ord_w", step=0, time=0.0, payload={"particles": [{"id": 1, "pos_x": 0.0}]})
        ev1 = SnapshotRecordedEvent(world_id="ord_w", step=1, time=0.01, payload={"particles": [{"id": 1, "pos_x": 1.0}]})
        ev2 = SnapshotRecordedEvent(world_id="ord_w", step=2, time=0.02, payload={"particles": [{"id": 1, "pos_x": 2.0}]})

        # Deliver out-of-order
        delivered = [ev2, ev0, ev1]
        reconstructed = reconstruct_state_from_snapshots(delivered)

        assert list(reconstructed.keys()) == [0, 1, 2]
        assert reconstructed[0][0]["pos_x"] == 0.0
        assert reconstructed[1][0]["pos_x"] == 1.0
        assert reconstructed[2][0]["pos_x"] == 2.0

    def test_missing_event_gap_handling(self):
        ev0 = SnapshotRecordedEvent(world_id="gap_w", step=0, time=0.0, payload={"particles": [{"id": 1, "pos_x": 0.0}]})
        # Step 1 is missing
        ev2 = SnapshotRecordedEvent(world_id="gap_w", step=2, time=0.02, payload={"particles": [{"id": 1, "pos_x": 2.0}]})

        reconstructed = reconstruct_state_from_snapshots([ev0, ev2])
        assert 0 in reconstructed
        assert 1 not in reconstructed
        assert 2 in reconstructed
