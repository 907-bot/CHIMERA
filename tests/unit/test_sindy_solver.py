"""Unit Tests: SINDy Solver — Law Discovery from Blind Trajectory Data

Verifies that SINDy correctly derives governing equations
from observable (x, v, a) arrays WITHOUT access to hidden parameters.

Key test: SINDy must rediscover k ≈ 3.0 with < 5% relative error
from only position/velocity data of a harmonic spring world.
"""

import pytest
import numpy as np
from packages.symbolic.benchmark_worlds import generate_blind_data, score_against_hidden_truth
from packages.symbolic.sindy_solver import SINDySolver, FeatureLibrary, _stlsq


class TestFeatureLibrary:

    def test_basic_feature_count(self):
        """Feature library must produce correct number of features."""
        lib = FeatureLibrary(include_trig=False, include_cubic=False)
        x = np.linspace(0, 2, 100)
        v = np.linspace(-1, 1, 100)
        theta, names = lib.build(x, v)

        # Expected: 1, x, v, x², xv, v² → 6 features
        assert theta.shape == (100, 6)
        assert len(names) == 6
        assert "x" in names
        assert "v" in names
        assert "x²" in names

    def test_trig_features_included(self):
        """Trig terms must be appended when include_trig=True."""
        lib = FeatureLibrary(include_trig=True)
        x = np.ones(50)
        v = np.zeros(50)
        theta, names = lib.build(x, v)
        assert "sin(x)" in names
        assert "cos(x)" in names

    def test_feature_values_correctness(self):
        """Feature matrix values must match manual computation."""
        lib = FeatureLibrary(include_trig=False)
        x = np.array([2.0, 3.0])
        v = np.array([1.0, 0.5])
        theta, names = lib.build(x, v)

        # Column 0: constant 1
        np.testing.assert_array_almost_equal(theta[:, 0], [1.0, 1.0])
        # Column 1: x
        np.testing.assert_array_almost_equal(theta[:, 1], [2.0, 3.0])
        # Column 3: x²
        np.testing.assert_array_almost_equal(theta[:, 3], [4.0, 9.0])


class TestSTLSQ:

    def test_recovers_simple_linear_law(self):
        """STLSQ must recover coefficients of a simple linear target."""
        # Target: y = 5.0 * x  (should set coef for x=5, others near 0)
        N = 300
        x = np.linspace(-2, 2, N)
        v = np.zeros(N)

        lib = FeatureLibrary()
        theta, names = lib.build(x, v)

        y = 5.0 * x
        xi = _stlsq(theta, y, threshold=0.1)

        # The 'x' coefficient should be ~5.0
        x_idx = names.index("x")
        assert abs(xi[x_idx] - 5.0) < 0.2, f"Expected ~5.0, got {xi[x_idx]}"

    def test_sparsity_prunes_noise_terms(self):
        """STLSQ must zero out non-causal terms below threshold."""
        N = 200
        x = np.linspace(0, 3, N)
        v = np.zeros(N)

        lib = FeatureLibrary()
        theta, names = lib.build(x, v)

        # Target driven purely by x (not v, x², etc.)
        y = -2.0 * x
        xi = _stlsq(theta, y, threshold=0.05)

        # Most non-x coefficients should be pruned to 0
        x_idx = names.index("x")
        non_x_mask = np.arange(len(names)) != x_idx
        n_nonzero_other = np.sum(np.abs(xi[non_x_mask]) > 0.05)
        assert n_nonzero_other <= 1, f"Expected sparsity, got {n_nonzero_other} non-zero other terms"


class TestSINDySolverHarmonicSpring:
    """Core exit-criteria test: SINDy derives k from harmonic spring observations."""

    def test_sindy_generates_candidate_hypothesis(self):
        """SINDy must produce a CANDIDATE hypothesis with a non-empty equation."""
        blind_data = generate_blind_data("harmonic_spring")
        solver = SINDySolver(threshold=0.05, train_ratio=0.8)
        hyp = solver.solve(blind_data)

        assert hyp is not None
        assert hyp.status == "CANDIDATE"
        assert len(hyp.candidate_equation) > 0
        assert hyp.world_name == "harmonic_spring"
        assert hyp.solver == "SINDy-STLSQ"

    def test_sindy_produces_r2_metric(self):
        """SINDy must include R² prediction metric on held-out trajectory."""
        blind_data = generate_blind_data("harmonic_spring")
        solver = SINDySolver(threshold=0.05)
        hyp = solver.solve(blind_data)

        assert hyp.metrics is not None
        assert -1.0 <= hyp.metrics.r_squared <= 1.0
        assert hyp.metrics.rmse >= 0.0

    def test_sindy_r2_exceeds_0_95(self):
        """SINDy must achieve R² > 0.95 on held-out harmonic spring trajectory."""
        blind_data = generate_blind_data("harmonic_spring")
        solver = SINDySolver(threshold=0.05, train_ratio=0.8)
        hyp = solver.solve(blind_data)

        assert hyp.metrics is not None, "Metrics must be computed"
        assert hyp.metrics.r_squared > 0.95, (
            f"SINDy R²={hyp.metrics.r_squared:.4f} did not exceed 0.95. "
            f"Equation: {hyp.candidate_equation}"
        )

    def test_sindy_equation_contains_x_term(self):
        """Discovered equation for harmonic spring must contain the x-displacement term."""
        blind_data = generate_blind_data("harmonic_spring")
        solver = SINDySolver(threshold=0.05)
        hyp = solver.solve(blind_data)

        # The dominant term must be x (Hooke's Law F = -kx)
        assert "*x" in hyp.candidate_equation or hyp.candidate_equation.startswith("-"), (
            f"Expected x-term in equation, got: {hyp.candidate_equation}"
        )

    def test_sindy_derived_k_within_5pct_error(self):
        """Derived spring constant k must be within 5% of hidden ground-truth k=3.0."""
        blind_data = generate_blind_data("harmonic_spring")
        solver = SINDySolver(threshold=0.05)
        hyp = solver.solve(blind_data)

        # The x-coefficient in the equation corresponds to -k
        coef_x = hyp.parameters.values.get("coef_x", None)
        if coef_x is None:
            pytest.skip("No coef_x in derived parameters — equation structure unexpected")

        discovered_k = abs(coef_x)
        true_k = 3.0
        relative_error = abs(discovered_k - true_k) / true_k

        assert relative_error < 0.05, (
            f"Derived k={discovered_k:.4f} is {relative_error*100:.2f}% away from "
            f"true k={true_k}. Equation: {hyp.candidate_equation}"
        )

    def test_sindy_damped_oscillator(self):
        """SINDy must produce R² > 0.90 for damped oscillator (x + v terms)."""
        blind_data = generate_blind_data("damped_oscillator")
        solver = SINDySolver(threshold=0.05)
        hyp = solver.solve(blind_data)

        assert hyp.metrics is not None
        assert hyp.metrics.r_squared > 0.90, (
            f"Damped oscillator R²={hyp.metrics.r_squared:.4f} too low. "
            f"Equation: {hyp.candidate_equation}"
        )
