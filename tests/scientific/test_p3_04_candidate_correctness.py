"""P3-04 — Candidate Equation Correctness

Compares discovered mathematical equations against reference physical laws.
Verifies equation terms, signs, coefficient magnitudes, and structural validity.
"""

import pytest
from packages.symbolic.benchmark_worlds import generate_blind_data
from packages.symbolic.sindy_solver import SINDySolver


class TestCandidateEquationCorrectness:
    """Scientific verification of discovered governing equations."""

    def test_harmonic_oscillator_equation_structure(self):
        """Hooke's Law: ẍ = -k*x (with k ≈ 3.0). Sign must be negative, term must be 'x'."""
        blind_data = generate_blind_data("harmonic_spring")
        solver = SINDySolver(threshold=0.05)
        hyp = solver.solve(blind_data)

        # 1. Active term check: 'coef_x' must be present
        assert "coef_x" in hyp.parameters.values
        coef_x = hyp.parameters.values["coef_x"]

        # 2. Sign check: must be negative (restoring force)
        assert coef_x < 0.0, f"Hooke's law restoring force must be negative, got {coef_x}"

        # 3. Magnitude check: true k=3.0, expected in [-3.2, -2.8]
        assert -3.2 <= coef_x <= -2.8, f"Coefficient out of range: {coef_x}"

        # 4. Spurious terms check: quadratic and cubic terms should be zeroed out
        assert "coef_x²" not in hyp.parameters.values
        assert "coef_v²" not in hyp.parameters.values

    def test_damped_oscillator_equation_structure(self):
        """Damped oscillator: ẍ = -k*x - b*v (k ≈ 2.5, b ≈ 0.3). Both terms negative."""
        blind_data = generate_blind_data("damped_oscillator")
        solver = SINDySolver(threshold=0.05)
        hyp = solver.solve(blind_data)

        assert "coef_x" in hyp.parameters.values
        assert "coef_v" in hyp.parameters.values

        coef_x = hyp.parameters.values["coef_x"]
        coef_v = hyp.parameters.values["coef_v"]

        # Both coefficients must be negative (restoring and damping)
        assert coef_x < 0.0, f"Restoring force must be negative, got {coef_x}"
        assert coef_v < 0.0, f"Damping force must be negative, got {coef_v}"

        # Magnitudes: k ≈ 2.5 in [-2.7, -2.3], b ≈ 0.3 in [-0.5, -0.1]
        assert -2.8 <= coef_x <= -2.2
        assert -0.5 <= coef_v <= -0.1
