"""P3-13 — SINDy Threshold Sensitivity

Tests STLSQ threshold sensitivity across low, medium, and high thresholds:
- Low threshold: includes more terms (potential overfitting / small noise terms)
- Optimal/Medium threshold: parsimonious recovery of true terms
- High threshold: over-pruning of terms
"""

import pytest
from packages.symbolic.benchmark_worlds import generate_blind_data
from packages.symbolic.sindy_solver import SINDySolver


class TestSINDyThresholdSensitivity:
    """Scientific test suite evaluating STLSQ sparsity threshold impact."""

    def test_threshold_scaling_effects(self):
        blind_data = generate_blind_data("harmonic_spring")

        # 1. Medium threshold (optimal): recovers purely linear x term
        solver_med = SINDySolver(threshold=0.05)
        hyp_med = solver_med.solve(blind_data)
        assert "coef_x" in hyp_med.parameters.values
        assert hyp_med.metrics.r_squared > 0.99

        # 2. High threshold (over-pruned): threshold=5.0 should prune all terms
        solver_high = SINDySolver(threshold=5.0)
        hyp_high = solver_high.solve(blind_data)
        assert len(hyp_high.parameters.values) == 0 or hyp_high.candidate_equation == "0"

        # 3. Low threshold (permissive): retains core term
        solver_low = SINDySolver(threshold=0.001)
        hyp_low = solver_low.solve(blind_data)
        assert "coef_x" in hyp_low.parameters.values
        assert hyp_low.metrics.r_squared > 0.99
