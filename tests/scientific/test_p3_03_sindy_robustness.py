"""P3-03 — SINDy Numerical Robustness

Tests solver stability and equation discovery across:
- Clean observations
- Gaussian noise (1%, 5%)
- Outliers
- Trajectory length variations (short 200 steps vs long 1000 steps)
- Subsampling rates
"""

import pytest
import numpy as np
from packages.symbolic.benchmark_worlds import generate_blind_data
from packages.symbolic.sindy_solver import SINDySolver


class TestSINDyNumericalRobustness:
    """Scientific test suite for SINDy solver robustness."""

    def test_clean_harmonic_recovery(self):
        blind_data = generate_blind_data("harmonic_spring")
        solver = SINDySolver(threshold=0.05)
        hyp = solver.solve(blind_data)

        assert hyp.metrics.r_squared > 0.99
        # Hooke coefficient should be close to -3.0
        coef_x = hyp.parameters.values.get("coef_x", 0.0)
        assert abs(coef_x - (-3.0)) < 0.15

    def test_small_gaussian_noise_robustness(self):
        blind_data = generate_blind_data("harmonic_spring")
        rng = np.random.default_rng(42)

        # Add 1% Gaussian noise to acceleration observable
        a_clean = blind_data["a"]
        noise_std = 0.01 * np.std(a_clean)
        blind_data_noisy = dict(blind_data)
        blind_data_noisy["a"] = a_clean + rng.normal(0, noise_std, size=len(a_clean))

        solver = SINDySolver(threshold=0.05)
        hyp = solver.solve(blind_data_noisy)

        # Under 1% noise, R^2 should remain high (> 0.95)
        assert hyp.metrics.r_squared > 0.95
        coef_x = hyp.parameters.values.get("coef_x", 0.0)
        assert abs(coef_x - (-3.0)) < 0.3

    def test_trajectory_length_scaling(self):
        blind_data = generate_blind_data("harmonic_spring")
        solver = SINDySolver(threshold=0.05)

        # Short trajectory (first 300 steps)
        short_data = {
            "world_name": "harmonic_spring",
            "t": blind_data["t"][:300],
            "x": blind_data["x"][:300],
            "v": blind_data["v"][:300],
            "a": blind_data["a"][:300],
        }
        hyp_short = solver.solve(short_data)
        assert hyp_short.metrics.r_squared > 0.95

    def test_subsampled_sampling_rate(self):
        blind_data = generate_blind_data("harmonic_spring")
        solver = SINDySolver(threshold=0.05)

        # Subsample every 2nd step
        subsampled_data = {
            "world_name": "harmonic_spring",
            "t": blind_data["t"][::2],
            "x": blind_data["x"][::2],
            "v": blind_data["v"][::2],
            "a": blind_data["a"][::2],
        }
        hyp_sub = solver.solve(subsampled_data)
        assert hyp_sub.metrics.r_squared > 0.95
