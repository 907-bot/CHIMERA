"""P3-15 — Scientific False-Discovery Test

Verifies that when given unphysical, contradictory, or pure white-noise data,
the discovery and validation system rejects the hypothesis (FALSIFIED or low R² CANDIDATE)
and never falsely proclaims it VALIDATED.
"""

import pytest
import numpy as np
from packages.symbolic.sindy_solver import SINDySolver
from packages.symbolic.hypothesis import Hypothesis, HypothesisParameters, PredictionMetrics


class TestFalseDiscoveryFalsification:
    """Scientific test suite validating hypothesis rejection on non-physical data."""

    def test_pure_white_noise_rejection(self):
        """Uncorrelated Gaussian white noise contains no governing ODE; validation must fail."""
        N = 500
        rng = np.random.default_rng(999)
        noise_data = {
            "world_name": "noise_world",
            "t": np.linspace(0, 10, N),
            "x": rng.normal(0, 1.0, size=N),
            "v": rng.normal(0, 1.0, size=N),
            "a": rng.normal(0, 5.0, size=N),  # Uncorrelated acceleration
        }

        solver = SINDySolver(threshold=0.05)
        hyp = solver.solve(noise_data)

        # On held-out test data, R^2 must be poor (< 0.50)
        assert hyp.metrics.r_squared < 0.50

        # Validate with standard threshold=0.99 must transition to FALSIFIED
        evaluated_hyp = hyp.validate(hyp.metrics, threshold=0.99)
        assert evaluated_hyp.status == "FALSIFIED"
        assert evaluated_hyp.falsification_evidence is not None

    def test_contradictory_physics_rejection(self):
        """Linearly increasing acceleration uncorrelated with position/velocity."""
        N = 400
        t = np.linspace(0, 4, N)
        bad_data = {
            "world_name": "bad_physics",
            "t": t,
            "x": np.sin(t),
            "v": np.cos(t),
            "a": np.exp(t),  # Exponential acceleration unrepresented in default linear/quadratic library
        }

        solver = SINDySolver(threshold=0.05)
        hyp = solver.solve(bad_data)

        evaluated = hyp.validate(hyp.metrics, threshold=0.99)
        assert evaluated.status == "FALSIFIED"
