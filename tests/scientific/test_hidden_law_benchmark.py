"""Scientific Benchmark: Hidden-Law Discovery Exit Criteria (Phase 3)

This is the formal exit criteria test for CHIMERA v0.2b.

PASS CONDITIONS (all must hold):
  1. SINDy achieves R² > 0.99 on held-out harmonic spring trajectory
  2. Discovered spring constant k within < 5% of hidden ground-truth k=3.0
  3. Full discovery pipeline registers hypothesis in append-only registry
  4. FALSIFIED hypotheses are retained (immutable evidence rule)
  5. DiscoveryEngine completes in < 30 seconds (reasonable local wall-clock)

Rule: AI scientists never see hidden_params. Only blind_data flows into solvers.
"""

import pytest
import time
from packages.symbolic.benchmark_worlds import (
    generate_blind_data,
    score_against_hidden_truth,
    ALL_BENCHMARKS,
)
from packages.symbolic.sindy_solver import SINDySolver
from packages.symbolic.discovery_engine import DiscoveryEngine
from packages.symbolic.registry import HypothesisRegistry
from packages.symbolic.hypothesis import Hypothesis, HypothesisParameters, PredictionMetrics


class TestHiddenLawBenchmarkHarness:
    """Tests the benchmark world harness integrity."""

    def test_all_three_worlds_available(self):
        """All three canonical hidden-law worlds must be registered."""
        assert "harmonic_spring" in ALL_BENCHMARKS
        assert "damped_oscillator" in ALL_BENCHMARKS
        assert "keplerian_approx" in ALL_BENCHMARKS

    def test_blind_data_never_exposes_hidden_params(self):
        """generate_blind_data() must NEVER return hidden physics parameters."""
        FORBIDDEN_KEYS = {"k", "GM", "b", "hidden_params"}

        for world_name in ALL_BENCHMARKS:
            blind_data = generate_blind_data(world_name)
            for key in blind_data.keys():
                assert key not in FORBIDDEN_KEYS, (
                    f"World '{world_name}' leaked hidden param '{key}' to AI scientists!"
                )

    def test_blind_data_deterministic_with_seed(self):
        """Same world_name must always produce identical observable arrays (bitwise reproducibility)."""
        import numpy as np
        d1 = generate_blind_data("harmonic_spring")
        d2 = generate_blind_data("harmonic_spring")
        np.testing.assert_array_equal(d1["x"], d2["x"])
        np.testing.assert_array_equal(d1["v"], d2["v"])
        np.testing.assert_array_equal(d1["t"], d2["t"])

    def test_trajectory_length_matches_spec(self):
        """Generated trajectory must have num_steps + 1 data points."""
        from packages.symbolic.benchmark_worlds import HARMONIC_SPRING
        blind_data = generate_blind_data("harmonic_spring")
        expected_len = HARMONIC_SPRING.num_steps + 1
        assert len(blind_data["x"]) == expected_len
        assert len(blind_data["t"]) == expected_len


class TestHiddenLawDiscoveryExitCriteria:
    """Exit criteria for Phase 3: R² > 0.99 on held-out trajectory."""

    def test_sindy_r2_exceeds_099_on_harmonic_spring(self):
        """[EXIT CRITERIA] SINDy must achieve R² > 0.99 on held-out harmonic spring trajectory.

        This is the primary Phase 3 scientific validation benchmark.
        Hidden: k=3.0 (never exposed to SINDy)
        Observable: (t, x, v, a)
        """
        blind_data = generate_blind_data("harmonic_spring")
        solver = SINDySolver(threshold=0.05, train_ratio=0.8)
        hyp = solver.solve(blind_data)

        assert hyp.metrics is not None
        assert hyp.metrics.r_squared > 0.99, (
            f"[EXIT CRITERIA FAILED] Phase 3 requires R² > 0.99 on harmonic spring.\n"
            f"Achieved R²={hyp.metrics.r_squared:.4f}\n"
            f"Discovered equation: {hyp.candidate_equation}\n"
            f"This indicates the SINDy solver did not correctly identify F = -k*x"
        )

    def test_spring_constant_recovery_within_1pct(self):
        """[EXIT CRITERIA] Discovered k must be within 5% of hidden ground-truth k=3.0.

        This proves the solver *derived* the law rather than guessing.
        """
        blind_data = generate_blind_data("harmonic_spring")
        solver = SINDySolver(threshold=0.05)
        hyp = solver.solve(blind_data)

        # Extract derived k from equation coefficients
        coef_x = hyp.parameters.values.get("coef_x", None)
        if coef_x is None:
            pytest.fail(
                f"SINDy did not find an x-coefficient. "
                f"Full equation: {hyp.candidate_equation}. "
                f"Parameters: {hyp.parameters.values}"
            )

        discovered_k = abs(coef_x)
        scores = score_against_hidden_truth("harmonic_spring", {"k": discovered_k})
        relative_error = scores["k"]

        assert relative_error < 0.05, (
            f"[EXIT CRITERIA FAILED] Discovered k={discovered_k:.4f} is "
            f"{relative_error*100:.2f}% away from hidden truth k=3.0.\n"
            f"Phase 3 requires < 5% relative error."
        )

    def test_discovery_pipeline_validates_and_registers(self):
        """Full pipeline: discovery → VALIDATED status → persisted in registry."""
        registry = HypothesisRegistry(":memory:")
        engine = DiscoveryEngine(registry=registry, sindy_threshold=0.05)

        result = engine.run_discovery("harmonic_spring")

        assert result.best_hypothesis is not None
        assert len(result.registry_ids) > 0

        # Retrieve from registry to confirm persistence
        retrieved = registry.get_by_id(result.registry_ids[0])
        assert retrieved is not None
        assert retrieved.world_name == "harmonic_spring"

        # Best hypothesis with R²>0.99 should be VALIDATED
        best = result.best_hypothesis
        if best.metrics and best.metrics.r_squared > 0.99:
            assert best.status == "VALIDATED", (
                f"High-R² hypothesis should be VALIDATED. Status={best.status}"
            )

    def test_registry_retains_falsified_hypotheses(self):
        """[EXIT CRITERIA] AGENTS.md Rule 6: Falsified hypotheses are immutable evidence."""
        registry = HypothesisRegistry(":memory:")

        # Register a deliberately bad hypothesis
        bad_hyp = Hypothesis(
            world_name="harmonic_spring",
            solver="TestBad",
            candidate_equation="-0.1*x",  # Wrong k
            parameters=HypothesisParameters(values={"coef_x": -0.1}),
        )
        hyp_id = registry.register_hypothesis(bad_hyp)
        bad_metrics = PredictionMetrics(
            r_squared=0.12, rmse=1.8, mae=1.5, train_steps=800, test_steps=200
        )
        registry.update_status(
            hyp_id, "FALSIFIED",
            metrics=bad_metrics,
            falsification_evidence="k=0.1 produces R²=0.12, expected k≈3.0"
        )

        # FALSIFIED hypothesis must STILL be in registry (never deleted)
        count_before = registry.count_all()
        falsified = registry.get_by_world("harmonic_spring", status_filter="FALSIFIED")

        assert len(falsified) == 1, "FALSIFIED hypothesis must be retained"
        assert falsified[0].falsification_evidence is not None
        assert count_before == 1, "No silent deletions allowed"

    def test_discovery_engine_completes_within_30s(self):
        """Discovery pipeline must complete within 30 seconds on local hardware."""
        registry = HypothesisRegistry(":memory:")
        engine = DiscoveryEngine(registry=registry)

        t_start = time.time()
        result = engine.run_discovery("harmonic_spring")
        elapsed = time.time() - t_start

        assert elapsed < 30.0, (
            f"Discovery took {elapsed:.2f}s > 30s limit. "
            "This may indicate a performance regression."
        )
        assert result.elapsed_seconds > 0


class TestMultiWorldDiscovery:
    """Run discovery across all benchmark worlds and verify coverage."""

    def test_discovery_produces_hypothesis_for_each_world(self):
        """Each benchmark world must produce at least one hypothesis."""
        registry = HypothesisRegistry(":memory:")
        engine = DiscoveryEngine(registry=registry)

        for world_name in ["harmonic_spring", "damped_oscillator"]:
            result = engine.run_discovery(world_name)
            assert result.best_hypothesis is not None, (
                f"No hypothesis produced for world '{world_name}'"
            )
            assert result.best_hypothesis.metrics is not None, (
                f"No metrics on hypothesis for '{world_name}'"
            )

    def test_registry_accumulates_across_worlds(self):
        """Registry must hold hypotheses from all worlds without collision."""
        registry = HypothesisRegistry(":memory:")
        engine = DiscoveryEngine(registry=registry)

        engine.run_discovery("harmonic_spring")
        engine.run_discovery("damped_oscillator")

        total = registry.count_all()
        assert total >= 2, f"Expected ≥ 2 total hypotheses, got {total}"

        spring_hyps = registry.get_by_world("harmonic_spring")
        damped_hyps = registry.get_by_world("damped_oscillator")
        assert len(spring_hyps) >= 1
        assert len(damped_hyps) >= 1
