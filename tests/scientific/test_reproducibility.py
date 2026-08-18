"""Scientific Benchmark: 100 Consecutive Bitwise Reproducibility Verification"""

import pytest
from packages.core.models import WorldConfig
from packages.core.serialization import hash_world_state
from packages.physics.engine import DeterministicEngine


def test_bitwise_reproducibility_100_runs():
    """Verify that 100 independent simulation runs with the same seed yield identical trajectory hashes."""
    seed = 42
    steps = 200
    config = WorldConfig(seed=seed, num_particles=15, dt=0.01)

    # Initial baseline run
    engine_baseline = DeterministicEngine(config=config)
    history_baseline = engine_baseline.run(steps)
    baseline_hash = hash_world_state(history_baseline[-1])

    # Run 100 consecutive verification iterations
    for i in range(100):
        engine = DeterministicEngine(config=config)
        history = engine.run(steps)
        current_hash = hash_world_state(history[-1])

        assert current_hash == baseline_hash, f"Iteration {i} failed bitwise reproducibility!"


def test_seed_sensitivity():
    """Verify that different random seeds produce different deterministic world histories."""
    config1 = WorldConfig(seed=42, num_particles=10)
    config2 = WorldConfig(seed=43, num_particles=10)

    engine1 = DeterministicEngine(config=config1)
    engine2 = DeterministicEngine(config=config2)

    history1 = engine1.run(100)
    history2 = engine2.run(100)

    hash1 = hash_world_state(history1[-1])
    hash2 = hash_world_state(history2[-1])

    assert hash1 != hash2, "Different seeds produced identical states!"
