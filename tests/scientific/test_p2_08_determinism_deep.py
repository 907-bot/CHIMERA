"""P2-08 — Deterministic Reproducibility

Verifies bitwise determinism across multiple independent runs with identical seed,
and verifies sensitivity to differing seeds (Seed 42 != Seed 43).
"""

import pytest
from packages.core.models import WorldConfig, WorldState
from packages.core.serialization import hash_world_state
from packages.physics.engine import DeterministicEngine
from packages.physics.energy import EnergyMetrics


class TestDeterministicReproducibilityDeep:
    """Test suite for bitwise determinism and seed sensitivity."""

    def test_multi_run_bitwise_identity(self):
        """Runs 1 through 5 with seed=42 must produce bitwise identical states and hashes."""
        config = WorldConfig(world_id="det_world", seed=42, num_particles=10, dt=0.01)

        runs = []
        hashes = []
        for _ in range(5):
            engine = DeterministicEngine(config=config)
            history = engine.run(200)
            final_s = history[-1]
            runs.append(final_s)
            hashes.append(hash_world_state(final_s))

        # All 5 hashes must be bitwise identical
        assert len(set(hashes)) == 1, f"Determinism failure: distinct hashes produced: {hashes}"

        # Check exact particle coordinates
        ref_state = runs[0]
        for run_idx, state in enumerate(runs[1:], start=2):
            assert ref_state.step == state.step
            assert ref_state.time == state.time
            for p_ref, p_curr in zip(ref_state.particles, state.particles):
                assert p_ref.id == p_curr.id
                assert p_ref.position.x == p_curr.position.x
                assert p_ref.position.y == p_curr.position.y
                assert p_ref.velocity.x == p_curr.velocity.x
                assert p_ref.velocity.y == p_curr.velocity.y

    def test_seed_sensitivity(self):
        """Seed 42 != Seed 43: differing seeds must yield different trajectories."""
        cfg1 = WorldConfig(world_id="w42", seed=42, num_particles=5, dt=0.01)
        cfg2 = WorldConfig(world_id="w43", seed=43, num_particles=5, dt=0.01)

        eng1 = DeterministicEngine(cfg1)
        eng2 = DeterministicEngine(cfg2)

        s1 = eng1.run(100)[-1]
        s2 = eng2.run(100)[-1]

        h1 = hash_world_state(s1)
        h2 = hash_world_state(s2)

        assert h1 != h2, "Seed sensitivity failure: seed=42 and seed=43 produced identical hashes!"
