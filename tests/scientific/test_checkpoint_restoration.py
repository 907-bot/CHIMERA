"""Scientific Benchmark: Checkpoint State Restoration Bitwise Equality"""

import pytest
from packages.core.models import WorldConfig
from packages.core.serialization import hash_world_state
from packages.physics.engine import DeterministicEngine


def test_checkpoint_restoration_bitwise_equality():
    """Verify that restoring a simulation state snapshot at tick T=500 and stepping to T=1000 produces bitwise identical results to a continuous T=0 to T=1000 run."""
    config = WorldConfig(seed=777, num_particles=12, dt=0.01)

    # 1. Continuous run T=0 -> T=1000
    engine_continuous = DeterministicEngine(config=config)
    history_continuous = engine_continuous.run(1000)
    hash_continuous_500 = hash_world_state(history_continuous[500])
    hash_continuous_1000 = hash_world_state(history_continuous[1000])

    # 2. Segment 1: T=0 -> T=500
    engine_checkpoint = DeterministicEngine(config=config)
    history_seg1 = engine_checkpoint.run(500)
    checkpoint_state_500 = history_seg1[500]

    assert hash_world_state(checkpoint_state_500) == hash_continuous_500

    # 3. Instantiate NEW fresh engine and restore state at step 500
    engine_restored = DeterministicEngine(config=config)
    engine_restored.restore_state(checkpoint_state_500)

    # 4. Segment 2: step remaining 500 steps (T=500 -> T=1000)
    history_seg2 = engine_restored.run(500)
    hash_restored_1000 = hash_world_state(history_seg2[-1])

    # 5. Bitwise equality assertion
    assert hash_restored_1000 == hash_continuous_1000, "Restored state trajectory diverged from continuous trajectory!"
