"""Scientific Benchmark: Observatory Query Performance Latency Criterion (< 50ms)"""

import time
import pytest
from packages.core.models import WorldConfig
from packages.physics.engine import DeterministicEngine
from packages.observatory.storage import ObservatoryStorageEngine


def test_observatory_100k_query_latency():
    """Verify DuckDB columnar observatory can store and query slices over 100,000 particle step records with < 50ms query latency."""
    storage = ObservatoryStorageEngine(":memory:")
    
    # 100 particles x 1,000 steps = 100,000 particle step records
    config = WorldConfig(world_id="perf_world", seed=999, num_particles=100, dt=0.01)
    engine = DeterministicEngine(config=config)

    print("\nSimulating 1,000 steps x 100 particles (100,000 particle-step records)...")
    history = engine.run(1000)

    # Record to DuckDB columnar store
    t_store_start = time.time()
    storage.store_trajectory(history)
    t_store_end = time.time()
    print(f"Stored 100,000 records in {(t_store_end - t_store_start)*1000:.2f} ms")

    assert storage.count_recorded_steps("perf_world") == 1001

    # Benchmark query latency for a 20-step frame slice (standard UI scrubber payload)
    t_query_start = time.time()
    slice_states = storage.query_trajectory_slice("perf_world", start_step=200, end_step=220)
    t_query_end = time.time()
    query_latency_ms = (t_query_end - t_query_start) * 1000.0

    print(f"Query Latency (20-step slice): {query_latency_ms:.2f} ms")
    assert len(slice_states) == 21

    # Performance Criterion: Query latency MUST be < 50ms
    assert query_latency_ms < 50.0, f"Query latency {query_latency_ms:.2f}ms exceeded exit criteria of 50ms!"

    storage.close()
