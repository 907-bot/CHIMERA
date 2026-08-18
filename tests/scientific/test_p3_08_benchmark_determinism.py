"""P3-08 — Benchmark Determinism

Verifies that benchmark worlds generate strictly reproducible blind observable arrays,
feature matrices, discovered equations, and prediction metrics across repeated runs.
"""

import pytest
import numpy as np
from packages.symbolic.benchmark_worlds import ALL_BENCHMARKS, generate_blind_data
from packages.symbolic.sindy_solver import SINDySolver


class TestBenchmarkDeterminism:
    """Scientific test suite for benchmark data generation determinism."""

    @pytest.mark.parametrize("world_name", ["harmonic_spring", "damped_oscillator", "keplerian_approx"])
    def test_benchmark_data_reproducibility(self, world_name):
        # Run benchmark twice
        data1 = generate_blind_data(world_name)
        data2 = generate_blind_data(world_name)

        # Check key array equality
        assert np.array_equal(data1["t"], data2["t"])
        assert np.array_equal(data1["x"], data2["x"])

        if "v" in data1:
            assert np.array_equal(data1["v"], data2["v"])
        if "a" in data1:
            assert np.array_equal(data1["a"], data2["a"])
        if "y" in data1:
            assert np.array_equal(data1["y"], data2["y"])

    def test_sindy_solution_determinism(self):
        data = generate_blind_data("harmonic_spring")
        solver = SINDySolver(threshold=0.05)

        hyp1 = solver.solve(data)
        hyp2 = solver.solve(data)

        assert hyp1.candidate_equation == hyp2.candidate_equation
        assert hyp1.metrics.r_squared == hyp2.metrics.r_squared
        assert hyp1.parameters.values == hyp2.parameters.values
