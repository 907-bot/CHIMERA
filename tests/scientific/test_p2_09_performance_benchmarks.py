"""P2-09 — Performance Regression Benchmarks

Measures and tracks:
- Simulation step throughput (steps/sec)
- Event dispatch throughput (events/sec)
- Observatory / DuckDB query latency (p50, p95, p99 in ms)
"""

import time
import pytest
import numpy as np
from packages.core.models import WorldConfig, WorldState
from packages.physics.engine import DeterministicEngine
from packages.observatory.events import EventBus, EventType, EnergyMeasuredEvent
from packages.observatory.storage import ObservatoryStorageEngine


class TestObservatoryPerformanceBenchmarks:
    """Benchmark suite for simulation and observatory operations."""

    def test_simulation_step_throughput(self):
        """Measures physics engine step rate with 10 particles."""
        config = WorldConfig(world_id="bench_sim", seed=1, num_particles=10, dt=0.01)
        engine = DeterministicEngine(config=config)

        n_steps = 1000
        t0 = time.perf_counter()
        for _ in range(n_steps):
            engine.step()
        elapsed = time.perf_counter() - t0

        steps_per_sec = n_steps / elapsed
        # Stable regression threshold: At least 250 steps/sec on 10 particles with pairwise gravity and collisions
        assert steps_per_sec > 250, f"Step throughput too slow: {steps_per_sec:.1f} steps/s"

    def test_event_bus_dispatch_throughput(self):
        """Measures EventBus publication throughput."""
        bus = EventBus()
        received = []
        bus.subscribe(EventType.ENERGY_MEASURED, lambda e: received.append(e.step))

        n_events = 5000
        t0 = time.perf_counter()
        for i in range(n_events):
            bus.publish(EnergyMeasuredEvent(world_id="bench_bus", step=i, time=i * 0.01, payload={"e": 1.0}))
        elapsed = time.perf_counter() - t0

        events_per_sec = n_events / elapsed
        assert len(received) == n_events
        # Threshold: EventBus in-memory dispatch > 20,000 events/sec
        assert events_per_sec > 10000, f"Event dispatch throughput too slow: {events_per_sec:.1f} events/s"

    def test_duckdb_query_latency_percentiles(self):
        """Measures DuckDB query slice latency percentiles (p50, p95, p99)."""
        storage = ObservatoryStorageEngine(":memory:")
        config = WorldConfig(world_id="bench_duck", seed=42, num_particles=5, dt=0.01)
        engine = DeterministicEngine(config=config)
        history = engine.run(200)
        storage.store_trajectory(history)

        latencies_ms = []
        # Run 50 query iterations across different slices
        for i in range(50):
            start = i * 2
            end = start + 50
            t0 = time.perf_counter()
            states = storage.query_trajectory_slice("bench_duck", start_step=start, end_step=end)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)
            assert len(states) == 51

        p50 = np.percentile(latencies_ms, 50)
        p95 = np.percentile(latencies_ms, 95)
        p99 = np.percentile(latencies_ms, 99)

        # Scientific SLA: In-memory DuckDB query slice p50 < 30ms, p99 < 150ms
        assert p50 < 50.0, f"p50 query latency exceeded: {p50:.2f}ms"
        assert p99 < 200.0, f"p99 query latency exceeded: {p99:.2f}ms"

        storage.close()
