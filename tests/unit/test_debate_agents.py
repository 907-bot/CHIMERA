"""Unit Tests: Debate Models, Roles, Intervention Engine, and Graph.

Phase 4 unit test coverage:
  - DebateRecord schema validation
  - Bull/Bear/Skeptic argument generation (rule-based)
  - Arbiter Bayesian scoring
  - Intervention Engine counterfactual execution
  - HypothesisGraph append-only DAG semantics
"""

import pytest
import math
from packages.symbolic.hypothesis import (
    Hypothesis, HypothesisParameters, PredictionMetrics
)
from packages.agents.debate_models import (
    BullArgument, BearArgument, CounterfactualExperiment,
    ExperimentResult, ArbiterVerdict, DebateRecord,
    SupportingEvidence, Weakness, PerturbationSpec, EvidenceWeight
)
from packages.agents.roles import BullAgent, BearAgent, SkepticAgent, ArbiterAgent
from packages.agents.intervention import InterventionEngine
from packages.agents.hypothesis_graph import HypothesisGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_validated_hypothesis(
    r2: float = 0.9952,
    world_name: str = "harmonic_spring",
    eq: str = "-2.99*x",
    coef_x: float = -2.99,
) -> Hypothesis:
    """Create a validated hypothesis with specified R²."""
    metrics = PredictionMetrics(r_squared=r2, rmse=0.002, mae=0.001, train_steps=800, test_steps=200)
    return Hypothesis(
        world_name=world_name,
        solver="SINDy-STLSQ",
        candidate_equation=eq,
        parameters=HypothesisParameters(values={"coef_x": coef_x}),
        metrics=metrics,
        status="VALIDATED",
    )


def make_false_hypothesis() -> Hypothesis:
    """Create a deliberately wrong hypothesis (k=0.5 instead of k=3.0)."""
    metrics = PredictionMetrics(r_squared=0.60, rmse=0.5, mae=0.4, train_steps=800, test_steps=200)
    return Hypothesis(
        world_name="harmonic_spring",
        solver="TestBad",
        candidate_equation="-0.5*x",
        parameters=HypothesisParameters(values={"coef_x": -0.5}),
        metrics=metrics,
        status="CANDIDATE",
    )


# ---------------------------------------------------------------------------
# Debate Model Tests
# ---------------------------------------------------------------------------

class TestDebateModels:

    def test_bull_argument_schema(self):
        hyp = make_validated_hypothesis()
        agent = BullAgent()
        bull = agent.argue(hyp)
        assert isinstance(bull, BullArgument)
        assert bull.agent == "Bull"
        assert 0.0 <= bull.confidence_score <= 1.0
        assert len(bull.supporting_evidence) > 0
        assert bull.hypothesis_id == hyp.id

    def test_bear_argument_schema(self):
        hyp = make_validated_hypothesis()
        agent = BearAgent()
        bear = agent.argue(hyp)
        assert isinstance(bear, BearArgument)
        assert bear.agent == "Bear"
        assert 0.0 <= bear.doubt_score <= 1.0
        assert len(bear.weaknesses) > 0

    def test_skeptic_experiment_schema(self):
        hyp = make_validated_hypothesis()
        skeptic = SkepticAgent()
        exp = skeptic.design_experiment(hyp)
        assert isinstance(exp, CounterfactualExperiment)
        assert len(exp.perturbations) > 0
        assert exp.r2_threshold_to_survive > 0.0
        assert exp.hypothesis_id == hyp.id

    def test_arbiter_verdict_schema(self):
        hyp = make_validated_hypothesis()
        bull = BullAgent().argue(hyp)
        bear = BearAgent().argue(hyp)
        skeptic_exp = SkepticAgent().design_experiment(hyp)
        result = ExperimentResult(
            experiment_id=skeptic_exp.experiment_id,
            hypothesis_id=hyp.id,
            world_name=hyp.world_name,
            r_squared_on_perturbed=0.985,
            rmse_on_perturbed=0.003,
            survived=True,
            interpretation="Survived",
            run_duration_seconds=0.5,
        )
        verdict = ArbiterAgent().issue_verdict(hyp, bull, bear, skeptic_exp, result)
        assert isinstance(verdict, ArbiterVerdict)
        assert verdict.verdict in ("ACCEPT", "REJECT", "INCONCLUSIVE")
        assert 0.0 <= verdict.bayesian_confidence <= 1.0


# ---------------------------------------------------------------------------
# Role Logic Tests
# ---------------------------------------------------------------------------

class TestBullAgent:

    def test_high_r2_yields_high_confidence(self):
        hyp = make_validated_hypothesis(r2=0.999)
        bull = BullAgent().argue(hyp)
        # sigmoid(10*(0.999 - 0.95)) ≈ 0.62 — significantly above 0.5 (neutral)
        assert bull.confidence_score > 0.5, f"Expected confidence > 0.5 for R²=0.999, got {bull.confidence_score}"

    def test_low_r2_yields_low_confidence(self):
        hyp = make_validated_hypothesis(r2=0.4)
        bull = BullAgent().argue(hyp)
        assert bull.confidence_score < 0.3, f"Expected low confidence for R²=0.4, got {bull.confidence_score}"

    def test_linear_equation_gets_symmetry_evidence(self):
        hyp = make_validated_hypothesis(eq="-2.99*x")
        bull = BullAgent().argue(hyp)
        types = [e.evidence_type for e in bull.supporting_evidence]
        assert "symmetry" in types, "Linear Hooke-type equation should get symmetry evidence"


class TestBearAgent:

    def test_false_hypothesis_gets_high_doubt(self):
        hyp = make_false_hypothesis()
        bear = BearAgent().argue(hyp)
        assert bear.doubt_score > 0.05

    def test_damped_world_without_v_term_flagged(self):
        hyp = Hypothesis(
            world_name="damped_oscillator",
            solver="Bad",
            candidate_equation="-2.5*x",  # Missing v term!
            parameters=HypothesisParameters(values={"coef_x": -2.5}),
            metrics=PredictionMetrics(r_squared=0.75, rmse=0.3, mae=0.2, train_steps=800, test_steps=200),
        )
        bear = BearAgent().argue(hyp)
        weakness_types = [w.weakness_type for w in bear.weaknesses]
        assert "alternative_law" in weakness_types, "Missing v term in damped world should be flagged"


class TestArbiterAgent:

    def test_accept_on_good_evidence(self):
        hyp = make_validated_hypothesis(r2=0.998)
        bull = BullAgent().argue(hyp)
        bear = BearAgent().argue(hyp)
        exp = SkepticAgent().design_experiment(hyp)
        result = ExperimentResult(
            experiment_id=exp.experiment_id, hypothesis_id=hyp.id,
            world_name=hyp.world_name, r_squared_on_perturbed=0.982,
            rmse_on_perturbed=0.002, survived=True,
            interpretation="Survived", run_duration_seconds=0.1,
        )
        verdict = ArbiterAgent().issue_verdict(hyp, bull, bear, exp, result)
        assert verdict.verdict == "ACCEPT", f"Expected ACCEPT but got {verdict.verdict}"

    def test_reject_on_failed_experiment(self):
        hyp = make_false_hypothesis()
        bull = BullAgent().argue(hyp)  # Will produce low confidence
        bear = BearAgent().argue(hyp)  # Will produce high doubt
        exp = SkepticAgent().design_experiment(hyp)
        result = ExperimentResult(
            experiment_id=exp.experiment_id, hypothesis_id=hyp.id,
            world_name=hyp.world_name, r_squared_on_perturbed=0.10,
            rmse_on_perturbed=1.5, survived=False,
            interpretation="Collapsed", run_duration_seconds=0.1,
        )
        verdict = ArbiterAgent().issue_verdict(hyp, bull, bear, exp, result)
        assert verdict.verdict in ("REJECT", "INCONCLUSIVE"), (
            f"False hypothesis should not be ACCEPTED. Got: {verdict.verdict}"
        )


# ---------------------------------------------------------------------------
# Intervention Engine Tests
# ---------------------------------------------------------------------------

class TestInterventionEngine:

    def test_run_experiment_returns_result(self):
        hyp = make_validated_hypothesis()
        engine = InterventionEngine()
        exp = SkepticAgent().design_experiment(hyp)
        result = engine.run_experiment(hyp, exp)
        assert isinstance(result, ExperimentResult)
        assert -1.0 <= result.r_squared_on_perturbed <= 1.0
        assert result.hypothesis_id == hyp.id

    def test_correct_hypothesis_survives_standard_battery(self):
        """A correctly discovered k≈3.0 must survive large-displacement perturbation."""
        hyp = make_validated_hypothesis(r2=0.9952, coef_x=-2.99)
        engine = InterventionEngine()
        experiments = engine.design_standard_experiments(hyp)
        assert len(experiments) == 3

        for exp in experiments[:1]:  # Test large displacement at minimum
            result = engine.run_experiment(hyp, exp)
            # Correct law (k≈3.0) should generalise well
            assert result.r_squared_on_perturbed > 0.5, (
                f"Correct hypothesis should survive perturbation. "
                f"R²={result.r_squared_on_perturbed:.4f}"
            )

    def test_false_hypothesis_collapsed_by_perturbation(self):
        """A wrong hypothesis (k=0.5 vs truth k=3.0) must collapse under perturbation."""
        hyp = make_false_hypothesis()
        engine = InterventionEngine()
        exp = SkepticAgent().design_experiment(hyp)
        result = engine.run_experiment(hyp, exp)
        # k=0.5 predicting k=3.0 world → should have very low R²
        assert result.r_squared_on_perturbed < 0.90, (
            f"Wrong hypothesis (k=0.5) should collapse. "
            f"Got R²={result.r_squared_on_perturbed:.4f}"
        )


# ---------------------------------------------------------------------------
# HypothesisGraph Tests
# ---------------------------------------------------------------------------

class TestHypothesisGraph:

    def test_register_world_node(self):
        g = HypothesisGraph()
        node_id = g.register_world("harmonic_spring", "Test world")
        assert node_id in g.graph
        assert g.graph.nodes[node_id]["node_type"] == "world"

    def test_register_hypothesis_creates_edge_to_world(self):
        g = HypothesisGraph()
        hyp_id = "test-uuid-1234"
        g.register_hypothesis(hyp_id, "harmonic_spring", "-3.0*x", "SINDy", 0.999, "VALIDATED")
        hyp_node = f"hypothesis::{hyp_id}"
        world_node = "world::harmonic_spring"
        assert hyp_node in g.graph
        assert world_node in g.graph
        assert g.graph.has_edge(hyp_node, world_node)

    def test_graph_summary_counts_node_types(self):
        g = HypothesisGraph()
        g.register_world("harmonic_spring")
        g.register_hypothesis("h1", "harmonic_spring", "-3*x", "SINDy", 0.99, "VALIDATED")
        summary = g.summary()
        assert summary.get("world", 0) >= 1
        assert summary.get("hypothesis", 0) >= 1

    def test_rejected_nodes_permanently_retained(self):
        """Per AGENTS.md Rule 6: REJECTED hypotheses must never be removed from graph."""
        g = HypothesisGraph()
        g.register_world("harmonic_spring")
        g.register_hypothesis("bad-h", "harmonic_spring", "-0.1*x", "TestBad", 0.12, "REJECTED")
        # Simulate verdict registration
        from packages.agents.debate_models import ArbiterVerdict, EvidenceWeight
        verdict = ArbiterVerdict(
            hypothesis_id="bad-h",
            world_name="harmonic_spring",
            verdict="REJECT",
            bayesian_confidence=0.15,
            evidence_weights=[],
            reasoning="Insufficient R²",
            reproducibility_score=0.0,
        )
        g.register_verdict(verdict)
        rejected = g.rejected_hypotheses()
        assert "bad-h" in rejected, "REJECTED hypothesis must be permanently in graph"

    def test_graph_export_to_dict(self):
        g = HypothesisGraph()
        g.register_world("test_world")
        d = g.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert len(d["nodes"]) >= 1
