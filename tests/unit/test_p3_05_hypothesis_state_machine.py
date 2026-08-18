"""P3-05 — Hypothesis Validation State Machine

Tests lifecycle transitions:
CANDIDATE -> VALIDATED
CANDIDATE -> FALSIFIED
Rejection of invalid transitions and immutability rules.
"""

import pytest
from packages.symbolic.hypothesis import Hypothesis, HypothesisParameters, PredictionMetrics
from packages.symbolic.registry import HypothesisRegistry


class TestHypothesisStateMachine:
    """Test suite for Hypothesis lifecycle transitions and registry state machine."""

    def test_candidate_to_validated_transition(self):
        hyp = Hypothesis(
            world_name="test_world",
            solver="SINDy",
            candidate_equation="-3.0*x",
            parameters=HypothesisParameters(values={"coef_x": -3.0}),
            status="CANDIDATE",
        )
        assert hyp.status == "CANDIDATE"

        # High R^2 (0.995) triggers VALIDATED state
        high_metrics = PredictionMetrics(
            r_squared=0.995,
            rmse=0.01,
            mae=0.008,
            train_steps=800,
            test_steps=200,
        )
        validated_hyp = hyp.validate(high_metrics, threshold=0.99)
        assert validated_hyp.status == "VALIDATED"
        assert validated_hyp.metrics.r_squared == 0.995

    def test_candidate_to_falsified_transition(self):
        hyp = Hypothesis(
            world_name="test_world",
            solver="SINDy",
            candidate_equation="-0.5*x",
            parameters=HypothesisParameters(values={"coef_x": -0.5}),
            status="CANDIDATE",
        )
        # Low R^2 (0.60) triggers FALSIFIED state with recorded evidence
        low_metrics = PredictionMetrics(
            r_squared=0.60,
            rmse=1.25,
            mae=0.95,
            train_steps=800,
            test_steps=200,
        )
        falsified_hyp = hyp.validate(low_metrics, threshold=0.99)
        assert falsified_hyp.status == "FALSIFIED"
        assert falsified_hyp.falsification_evidence is not None
        assert "R²=0.6000 < threshold=0.9900" in falsified_hyp.falsification_evidence

    def test_registry_rejects_invalid_status(self):
        registry = HypothesisRegistry(":memory:")
        hyp = Hypothesis(
            world_name="test_world",
            solver="SINDy",
            candidate_equation="-3.0*x",
            parameters=HypothesisParameters(values={"coef_x": -3.0}),
            status="CANDIDATE",
        )
        reg_id = registry.register_hypothesis(hyp)

        # Invalid transition should raise ValueError
        with pytest.raises(ValueError, match="Invalid status transition"):
            registry.update_status(reg_id, "NONEXISTENT_STATUS")

        registry.close()

    def test_update_nonexistent_hypothesis_raises_keyerror(self):
        registry = HypothesisRegistry(":memory:")
        with pytest.raises(KeyError):
            registry.update_status("nonexistent-uuid", "VALIDATED")
        registry.close()
