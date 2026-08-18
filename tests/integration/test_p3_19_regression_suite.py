"""P3-19 — Phase 3 Regression Suite

Consolidates Phase 3 symbolic discovery regression tests.
"""

import pytest
from packages.symbolic.benchmark_worlds import ALL_BENCHMARKS
from packages.symbolic.discovery_engine import DiscoveryEngine
from packages.symbolic.registry import HypothesisRegistry


class TestPhase3RegressionSuite:
    """Consolidated regression suite for Phase 3 components."""

    def test_full_symbolic_discovery_regression(self):
        reg = HypothesisRegistry(":memory:")
        eng = DiscoveryEngine(registry=reg)

        for world in ["harmonic_spring", "damped_oscillator"]:
            res = eng.run_discovery(world)
            assert res.best_hypothesis is not None
            assert res.best_hypothesis.metrics.r_squared > 0.95
            assert res.best_hypothesis.status in ("VALIDATED", "CANDIDATE")

            stored = reg.get_by_world(world)
            assert len(stored) >= 1

        reg.close()
