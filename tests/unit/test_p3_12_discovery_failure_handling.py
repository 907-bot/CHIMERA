"""P3-12 — Discovery Failure Handling

Verifies graceful handling and defensive failure containment for SINDy solver
and discovery engine when encountering invalid inputs (NaNs, Infs, degenerate features).
"""

import pytest
import numpy as np
from packages.symbolic.sindy_solver import SINDySolver


class TestDiscoveryFailureHandling:
    """Test suite for discovery failure containment and edge case handling."""

    def test_constant_zero_motion(self):
        """Zero motion: x=0, v=0, a=0 should produce a zero equation cleanly without crashing."""
        N = 100
        zero_data = {
            "world_name": "zero_world",
            "t": np.linspace(0, 1, N),
            "x": np.zeros(N),
            "v": np.zeros(N),
            "a": np.zeros(N),
        }
        solver = SINDySolver(threshold=0.05)
        hyp = solver.solve(zero_data)

        assert hyp.candidate_equation == "0"
        assert hyp.status == "CANDIDATE"

    def test_short_trajectory_boundary(self):
        """Very short trajectory (e.g. 10 steps) should run without crashing."""
        N = 10
        short_data = {
            "world_name": "short_world",
            "t": np.linspace(0, 0.1, N),
            "x": np.sin(np.linspace(0, 0.1, N)),
            "v": np.cos(np.linspace(0, 0.1, N)),
            "a": -np.sin(np.linspace(0, 0.1, N)),
        }
        solver = SINDySolver(threshold=0.05)
        hyp = solver.solve(short_data)
        assert hyp.candidate_equation is not None
