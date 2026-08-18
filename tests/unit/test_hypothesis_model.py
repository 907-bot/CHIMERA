"""Unit Tests: Hypothesis Data Model and Append-Only Registry

Tests:
  - Hypothesis creation, lifecycle (CANDIDATE → VALIDATED / FALSIFIED)
  - Registry append-only semantics (AGENTS.md Rule 6)
  - Falsified hypotheses are permanently retained — never deleted
"""

import pytest
from packages.symbolic.hypothesis import (
    Hypothesis,
    HypothesisParameters,
    PredictionMetrics,
)
from packages.symbolic.registry import HypothesisRegistry


# ---------------------------------------------------------------------------
# Hypothesis Model Tests
# ---------------------------------------------------------------------------

class TestHypothesisModel:

    def test_hypothesis_candidate_default_status(self):
        """Newly created hypothesis must default to CANDIDATE status."""
        hyp = Hypothesis(
            world_name="harmonic_spring",
            solver="TestSolver",
            candidate_equation="-3.0*x",
            parameters=HypothesisParameters(values={"coef_x": -3.0}),
        )
        assert hyp.status == "CANDIDATE"
        assert hyp.id is not None
        assert hyp.metrics is None

    def test_hypothesis_validate_above_threshold(self):
        """Hypothesis with R² ≥ 0.99 must be promoted to VALIDATED."""
        hyp = Hypothesis(
            world_name="harmonic_spring",
            solver="SINDy",
            candidate_equation="-2.99*x",
            parameters=HypothesisParameters(values={"coef_x": -2.99}),
        )
        metrics = PredictionMetrics(
            r_squared=0.9952,
            rmse=0.002,
            mae=0.001,
            train_steps=800,
            test_steps=200,
        )
        validated = hyp.validate(metrics, threshold=0.99)
        assert validated.status == "VALIDATED"
        assert validated.falsification_evidence is None
        assert validated.metrics is not None
        assert validated.metrics.r_squared == pytest.approx(0.9952)

    def test_hypothesis_validate_below_threshold(self):
        """Hypothesis with R² < 0.99 must be marked FALSIFIED with evidence string."""
        hyp = Hypothesis(
            world_name="harmonic_spring",
            solver="SINDy",
            candidate_equation="-1.5*x",
            parameters=HypothesisParameters(values={"coef_x": -1.5}),
        )
        bad_metrics = PredictionMetrics(
            r_squared=0.72,
            rmse=0.45,
            mae=0.30,
            train_steps=800,
            test_steps=200,
        )
        falsified = hyp.validate(bad_metrics, threshold=0.99)
        assert falsified.status == "FALSIFIED"
        assert falsified.falsification_evidence is not None
        assert "R²" in falsified.falsification_evidence

    def test_hypothesis_summary_string(self):
        """Summary string must include status, solver, world name, equation, and R²."""
        metrics = PredictionMetrics(r_squared=0.995, rmse=0.01, mae=0.005, train_steps=800, test_steps=200)
        hyp = Hypothesis(
            world_name="damped_oscillator",
            solver="SINDy",
            candidate_equation="-2.5*x + -0.3*v",
            parameters=HypothesisParameters(values={"coef_x": -2.5, "coef_v": -0.3}),
            metrics=metrics,
            status="VALIDATED",
        )
        summary = hyp.summary()
        assert "VALIDATED" in summary
        assert "SINDy" in summary
        assert "damped_oscillator" in summary


# ---------------------------------------------------------------------------
# Hypothesis Registry Tests (Append-Only Semantics)
# ---------------------------------------------------------------------------

class TestHypothesisRegistry:

    def _make_hyp(self, world_name: str = "harmonic_spring", solver: str = "SINDy") -> Hypothesis:
        return Hypothesis(
            world_name=world_name,
            solver=solver,
            candidate_equation="-3.0*x",
            parameters=HypothesisParameters(values={"coef_x": -3.0}),
        )

    def test_register_and_retrieve(self):
        """Registered hypothesis must be retrievable by ID."""
        reg = HypothesisRegistry(":memory:")
        hyp = self._make_hyp()
        reg_id = reg.register_hypothesis(hyp)
        retrieved = reg.get_by_id(reg_id)
        assert retrieved is not None
        assert retrieved.id == hyp.id
        assert retrieved.candidate_equation == "-3.0*x"
        assert retrieved.status == "CANDIDATE"

    def test_total_count_after_registration(self):
        """Registry must track total count including all lifecycle states."""
        reg = HypothesisRegistry(":memory:")
        reg.register_hypothesis(self._make_hyp("harmonic_spring"))
        reg.register_hypothesis(self._make_hyp("damped_oscillator"))
        assert reg.count_all() == 2

    def test_falsified_hypothesis_retained(self):
        """Per AGENTS.md Rule 6: FALSIFIED hypotheses MUST be retained permanently."""
        reg = HypothesisRegistry(":memory:")
        hyp = self._make_hyp()
        reg_id = reg.register_hypothesis(hyp)

        bad_metrics = PredictionMetrics(
            r_squared=0.55, rmse=0.99, mae=0.88, train_steps=800, test_steps=200
        )
        reg.update_status(
            reg_id,
            "FALSIFIED",
            metrics=bad_metrics,
            falsification_evidence="R²=0.55 failed threshold",
        )

        # Verify it's still in the registry (NOT deleted)
        retrieved = reg.get_by_id(reg_id)
        assert retrieved is not None
        assert retrieved.status == "FALSIFIED"
        assert retrieved.falsification_evidence is not None
        assert reg.count_all() == 1  # Still 1 record — immutable evidence

    def test_get_by_world_with_status_filter(self):
        """Registry must support filtering by world name and status."""
        reg = HypothesisRegistry(":memory:")

        h1 = self._make_hyp("harmonic_spring")
        h2 = self._make_hyp("harmonic_spring")
        h3 = self._make_hyp("damped_oscillator")

        id1 = reg.register_hypothesis(h1)
        id2 = reg.register_hypothesis(h2)
        reg.register_hypothesis(h3)

        # Promote h1 to VALIDATED
        metrics = PredictionMetrics(r_squared=0.995, rmse=0.01, mae=0.005, train_steps=800, test_steps=200)
        reg.update_status(id1, "VALIDATED", metrics=metrics)
        reg.update_status(id2, "FALSIFIED")

        validated = reg.get_by_world("harmonic_spring", status_filter="VALIDATED")
        falsified = reg.get_by_world("harmonic_spring", status_filter="FALSIFIED")
        all_spring = reg.get_by_world("harmonic_spring")

        assert len(validated) == 1
        assert len(falsified) == 1
        assert len(all_spring) == 2  # Both retained

    def test_no_duplicate_registration(self):
        """Registering the same hypothesis ID twice must raise an error."""
        reg = HypothesisRegistry(":memory:")
        hyp = self._make_hyp()
        reg.register_hypothesis(hyp)
        with pytest.raises(Exception):
            reg.register_hypothesis(hyp)  # Same UUID — must fail
