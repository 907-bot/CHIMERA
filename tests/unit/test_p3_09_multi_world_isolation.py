"""P3-09 — Multi-World Isolation

Verifies that executing benchmarks in different permutation orders yields
strictly independent results with no cross-world state contamination.
"""

import pytest
from packages.symbolic.discovery_engine import DiscoveryEngine
from packages.symbolic.registry import HypothesisRegistry


class TestMultiWorldIsolation:
    """Test suite for benchmark execution order independence and world isolation."""

    def test_order_independence(self):
        worlds = ["harmonic_spring", "damped_oscillator"]

        # Order 1: harmonic -> damped
        reg1 = HypothesisRegistry(":memory:")
        eng1 = DiscoveryEngine(registry=reg1)
        res1_harmonic = eng1.run_discovery("harmonic_spring")
        res1_damped = eng1.run_discovery("damped_oscillator")

        # Order 2: damped -> harmonic
        reg2 = HypothesisRegistry(":memory:")
        eng2 = DiscoveryEngine(registry=reg2)
        res2_damped = eng2.run_discovery("damped_oscillator")
        res2_harmonic = eng2.run_discovery("harmonic_spring")

        # Equations and metrics must be identical regardless of order
        assert res1_harmonic.best_hypothesis.candidate_equation == res2_harmonic.best_hypothesis.candidate_equation
        assert res1_harmonic.best_hypothesis.metrics.r_squared == res2_harmonic.best_hypothesis.metrics.r_squared

        assert res1_damped.best_hypothesis.candidate_equation == res2_damped.best_hypothesis.candidate_equation
        assert res1_damped.best_hypothesis.metrics.r_squared == res2_damped.best_hypothesis.metrics.r_squared

        reg1.close()
        reg2.close()

    def test_registry_world_partitioning(self):
        reg = HypothesisRegistry(":memory:")
        eng = DiscoveryEngine(registry=reg)

        eng.run_discovery("harmonic_spring")
        eng.run_discovery("damped_oscillator")

        harmonic_hyps = reg.get_by_world("harmonic_spring")
        damped_hyps = reg.get_by_world("damped_oscillator")

        assert len(harmonic_hyps) == 1
        assert len(damped_hyps) == 1
        assert harmonic_hyps[0].world_name == "harmonic_spring"
        assert damped_hyps[0].world_name == "damped_oscillator"

        reg.close()
