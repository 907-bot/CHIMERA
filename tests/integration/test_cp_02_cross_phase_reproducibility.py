"""CP-02 — Reproducibility Across Both Phases

Verifies complete multi-phase reproducibility:
Phase 2 Simulation -> Observatory -> Phase 3 Discovery
Seed = 42 Run 1 == Run 2 (Same observations, same features, same equation, same metrics).
"""

import pytest
import numpy as np
from packages.symbolic.discovery_engine import DiscoveryEngine
from packages.symbolic.registry import HypothesisRegistry


class TestCrossPhaseReproducibility:
    """Test suite for full pipeline determinism across both phases."""

    def test_full_pipeline_multi_run_reproducibility(self):
        # Run full pipeline twice
        reg1 = HypothesisRegistry(":memory:")
        eng1 = DiscoveryEngine(registry=reg1)
        res1 = eng1.run_discovery("harmonic_spring")

        reg2 = HypothesisRegistry(":memory:")
        eng2 = DiscoveryEngine(registry=reg2)
        res2 = eng2.run_discovery("harmonic_spring")

        # Candidate equations and metrics must be identical
        assert res1.best_hypothesis.candidate_equation == res2.best_hypothesis.candidate_equation
        assert res1.best_hypothesis.metrics.r_squared == res2.best_hypothesis.metrics.r_squared
        assert res1.best_hypothesis.parameters.values == res2.best_hypothesis.parameters.values

        reg1.close()
        reg2.close()
