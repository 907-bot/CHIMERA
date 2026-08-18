"""P2-03 — Checkpoint / Restore Verification

Deep verification of checkpoint serialization and deterministic state restoration.
Compares:
Run to step 500 -> Continue to step 1000
vs
Run to step 500 -> Checkpoint -> Restore -> Continue to step 1000

Asserts: Final A == Final B (Positions, Velocities, Hashes, Telemetry, Serialization)
"""

import pytest
import json
from packages.core.models import WorldConfig, WorldState
from packages.core.serialization import hash_world_state
from packages.physics.engine import DeterministicEngine
from packages.physics.energy import EnergyMetrics


class TestCheckpointRestoreDeepVerification:
    """Test suite for checkpoint serialization, restore, and exact future continuation."""

    def test_checkpoint_restore_reproduces_future_trajectory(self):
        config = WorldConfig(
            world_id="checkpoint_world",
            seed=101,
            num_particles=8,
            dt=0.01,
            integrator_type="verlet",
        )

        # Baseline: Run straight to 1000 steps
        engine_a = DeterministicEngine(config=config)
        history_a = engine_a.run(1000)
        final_state_a = history_a[-1]
        checkpoint_state = history_a[500]

        # Branch B: Run to 500, checkpoint, restore in a fresh engine, and continue to 1000
        engine_b = DeterministicEngine(config=config)
        history_b_first_half = engine_b.run(500)
        assert history_b_first_half[500].step == 500

        # Save checkpoint to JSON (simulating disk persistence)
        checkpoint_json = checkpoint_state.model_dump_json()

        # Restore from JSON in a fresh engine
        restored_state = WorldState.model_validate_json(checkpoint_json)
        engine_c = DeterministicEngine(config=config)
        engine_c.restore_state(restored_state)

        # Continue from restored state for remaining 500 steps (reaching step 1000)
        history_c = [engine_c.current_state]
        for _ in range(500):
            history_c.append(engine_c.step())

        final_state_c = history_c[-1]

        # Verification 1: Step & Time match
        assert final_state_a.step == 1000
        assert final_state_c.step == 1000
        assert abs(final_state_a.time - final_state_c.time) < 1e-9

        # Verification 2: Bitwise State Hashes Match
        hash_a = hash_world_state(final_state_a)
        hash_c = hash_world_state(final_state_c)
        assert hash_a == hash_c

        # Verification 3: Particle positions, velocities, and masses match exactly
        assert len(final_state_a.particles) == len(final_state_c.particles)
        for pa, pc in zip(final_state_a.particles, final_state_c.particles):
            assert pa.id == pc.id
            assert pa.mass == pc.mass
            assert pa.radius == pc.radius
            assert abs(pa.position.x - pc.position.x) < 1e-9
            assert abs(pa.position.y - pc.position.y) < 1e-9
            assert abs(pa.velocity.x - pc.velocity.x) < 1e-9
            assert abs(pa.velocity.y - pc.velocity.y) < 1e-9

        # Verification 4: Telemetry / Energy metrics match
        energy_a = EnergyMetrics.compute_all(final_state_a.particles)
        energy_c = EnergyMetrics.compute_all(final_state_c.particles)
        assert abs(energy_a["total_energy"] - energy_c["total_energy"]) < 1e-9
        assert abs(energy_a["kinetic_energy"] - energy_c["kinetic_energy"]) < 1e-9
        assert abs(energy_a["potential_energy"] - energy_c["potential_energy"]) < 1e-9

        # Verification 5: Serialized representations match exactly
        assert final_state_a.model_dump_json() == final_state_c.model_dump_json()

    def test_checkpoint_restore_rk4_integrator(self):
        config = WorldConfig(
            world_id="checkpoint_rk4",
            seed=202,
            num_particles=5,
            dt=0.01,
            integrator_type="rk4",
        )
        engine_a = DeterministicEngine(config=config)
        history_a = engine_a.run(200)

        # Run to 100, restore, run remaining 100
        engine_b = DeterministicEngine(config=config)
        engine_b.run(100)
        checkpoint_json = history_a[100].model_dump_json()

        engine_restored = DeterministicEngine(config=config)
        engine_restored.restore_state(WorldState.model_validate_json(checkpoint_json))
        history_restored = [engine_restored.current_state]
        for _ in range(100):
            history_restored.append(engine_restored.step())

        assert hash_world_state(history_a[-1]) == hash_world_state(history_restored[-1])
