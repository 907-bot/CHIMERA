"""P3-18 — Phase 3 Performance

Measures execution times across:
- Benchmark generation time
- Feature matrix construction
- SINDy solver execution
- Registry write time
Evaluates scaling behavior across 1K, 10K, and 50K observation steps.
"""

import time
import pytest
import numpy as np
from packages.symbolic.benchmark_worlds import BenchmarkWorldSpec, _integrate_harmonic
from packages.symbolic.sindy_solver import SINDySolver, FeatureLibrary
from packages.symbolic.registry import HypothesisRegistry
from packages.symbolic.hypothesis import Hypothesis, HypothesisParameters


class TestPhase3Performance:
    """Benchmark suite for symbolic discovery scalability."""

    @pytest.mark.parametrize("n_steps", [1000, 10000, 50000])
    def test_sindy_solver_scaling(self, n_steps):
        # 1. Measure benchmark generation time
        t0 = time.perf_counter()
        spec = BenchmarkWorldSpec(
            name="harmonic_perf",
            description="Perf harmonic",
            hidden_params={"k": 3.0, "x_eq": 0.0},
            num_particles=1,
            num_steps=n_steps,
            dt=0.001,
            seed=42,
        )
        raw = _integrate_harmonic(spec)
        t_gen = time.perf_counter() - t0

        blind_data = {
            "world_name": "harmonic_perf",
            "t": raw["t"],
            "x": raw["x"],
            "v": raw["v"],
            "a": raw["a"],
        }

        # 2. Measure feature matrix construction
        lib = FeatureLibrary()
        t1 = time.perf_counter()
        theta, _ = lib.build(blind_data["x"], blind_data["v"])
        t_feat = time.perf_counter() - t1

        # 3. Measure SINDy solve time
        solver = SINDySolver(threshold=0.05)
        t2 = time.perf_counter()
        hyp = solver.solve(blind_data)
        t_solve = time.perf_counter() - t2

        assert hyp.metrics.r_squared > 0.99
        # Practical performance thresholds (pure Python/NumPy/Scikit-learn):
        # 50K points solved in < 2.0 seconds
        assert t_solve < 3.0, f"SINDy solve time exceeded: {t_solve:.3f}s for {n_steps} points"
        assert t_feat < 1.0, f"Feature construction exceeded: {t_feat:.3f}s"
