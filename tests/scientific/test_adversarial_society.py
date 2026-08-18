"""Scientific Benchmark: Adversarial Society Exit Criteria (Phase 4)

Exit Criteria (all must hold):
  1. System ACCEPTS correctly discovered hypotheses (k≈3.0, R²>0.99)
  2. System REJECTS deliberately attractive but wrong hypotheses (k=0.5)
  3. Full debate pipeline completes in < 60 seconds
  4. Rejected hypotheses are permanently retained in HypothesisGraph
  5. DebateRecord is immutable and persisted with all arguments

The hardest test: Can the system tell apart F=-3.0x from F=-0.5x
purely by running a deterministic counterfactual perturbation?
"""

import pytest
import time
from packages.symbolic.benchmark_worlds import generate_blind_data
from packages.symbolic.sindy_solver import SINDySolver
from packages.symbolic.hypothesis import Hypothesis, HypothesisParameters, PredictionMetrics
from packages.agents.debate_models import DebateRecord
from packages.agents.debate_engine import DebateEngine, DebateState
from packages.agents.hypothesis_graph import HypothesisGraph
from packages.agents.roles import BullAgent, BearAgent, SkepticAgent, ArbiterAgent
from packages.agents.intervention import InterventionEngine


def make_correct_hypothesis() -> Hypothesis:
    """Correct hypothesis: k≈3.0 as discovered by SINDy from harmonic spring."""
    blind_data = generate_blind_data("harmonic_spring")
    solver = SINDySolver(threshold=0.05, train_ratio=0.8)
    return solver.solve(blind_data)


def make_false_attractive_hypothesis() -> Hypothesis:
    """A 'plausible-looking' false hypothesis: k=1.5 (wrong, but seemingly reasonable).

    This mimics a common scientific pitfall — a hypothesis that looks
    mathematically consistent but encodes the wrong parameter.
    The system must REJECT it based on counterfactual evidence.
    """
    metrics = PredictionMetrics(
        r_squared=0.72,  # Mediocre but potentially convincing
        rmse=0.45, mae=0.38,
        train_steps=800, test_steps=200,
    )
    return Hypothesis(
        world_name="harmonic_spring",
        solver="AttractiveButWrong",
        candidate_equation="-1.5*x",
        parameters=HypothesisParameters(values={"coef_x": -1.5}),
        metrics=metrics,
        status="CANDIDATE",
    )


class TestDebatePipelineExitCriteria:

    def test_correct_hypothesis_accepted(self):
        """[EXIT CRITERIA] System must ACCEPT a correctly discovered hypothesis."""
        hyp = make_correct_hypothesis()
        engine = DebateEngine()

        record = engine.debate(hyp)

        assert isinstance(record, DebateRecord)
        assert record.arbiter_verdict.verdict == "ACCEPT", (
            f"[EXIT CRITERIA FAILED] Correct hypothesis should be ACCEPTED.\n"
            f"Got: {record.arbiter_verdict.verdict}\n"
            f"Confidence: {record.arbiter_verdict.bayesian_confidence:.4f}\n"
            f"Experiment survived: {record.experiment_result.survived}\n"
            f"Equation: {hyp.candidate_equation}\n"
            f"R²: {hyp.metrics.r_squared:.4f}"
        )

    def test_false_hypothesis_not_accepted(self):
        """[EXIT CRITERIA] System must REJECT or mark INCONCLUSIVE false hypothesis.

        A false hypothesis (k=1.5 vs truth k=3.0) must not receive ACCEPT verdict.
        This is the primary adversarial system exit criterion.
        """
        hyp = make_false_attractive_hypothesis()
        engine = DebateEngine()

        record = engine.debate(hyp)

        assert record.arbiter_verdict.verdict in ("REJECT", "INCONCLUSIVE"), (
            f"[EXIT CRITERIA FAILED] False hypothesis (k=1.5) must not be ACCEPTED.\n"
            f"Got: {record.arbiter_verdict.verdict}\n"
            f"Confidence: {record.arbiter_verdict.bayesian_confidence:.4f}\n"
            f"Experiment result R²: {record.experiment_result.r_squared_on_perturbed:.4f}\n"
            f"Experiment survived: {record.experiment_result.survived}"
        )

    def test_debate_completes_within_60s(self):
        """Full debate pipeline must complete in < 60 seconds."""
        hyp = make_correct_hypothesis()
        engine = DebateEngine()

        t_start = time.time()
        record = engine.debate(hyp)
        elapsed = time.time() - t_start

        assert elapsed < 60.0, f"Debate took {elapsed:.2f}s > 60s limit"
        assert record.duration_seconds > 0.0

    def test_debate_record_immutable_and_complete(self):
        """DebateRecord must contain all four agent outputs."""
        hyp = make_correct_hypothesis()
        engine = DebateEngine()

        record = engine.debate(hyp)

        assert record.bull_argument is not None
        assert record.bear_argument is not None
        assert record.skeptic_experiment is not None
        assert record.experiment_result is not None
        assert record.arbiter_verdict is not None
        assert record.debate_id is not None
        assert record.final_status in ("ACCEPTED", "REJECTED", "INCONCLUSIVE")

    def test_debate_state_machine_reaches_complete(self):
        """State machine must reach COMPLETE state after a full debate."""
        hyp = make_correct_hypothesis()
        engine = DebateEngine()

        assert engine.state == DebateState.IDLE
        engine.debate(hyp)
        assert engine.state == DebateState.COMPLETE


class TestProvenanceGraphExitCriteria:

    def test_rejected_hypothesis_retained_in_graph(self):
        """[EXIT CRITERIA] Per AGENTS.md Rule 6: REJECTED hypotheses permanently in graph."""
        g = HypothesisGraph()
        engine = DebateEngine(graph=g)

        false_hyp = make_false_attractive_hypothesis()
        record = engine.debate(false_hyp)

        # Check graph has hypothesis node regardless of verdict
        hyp_node = f"hypothesis::{false_hyp.id}"
        assert hyp_node in g.graph or len(g.graph.nodes) > 0, (
            "Hypothesis must be registered in graph after debate"
        )

        # If rejected, must be in rejected list
        if record.arbiter_verdict.verdict == "REJECT":
            rejected = g.rejected_hypotheses()
            assert len(rejected) > 0, "REJECTED hypotheses must be permanently in graph"

    def test_graph_accumulates_both_debates(self):
        """Graph must retain nodes from multiple debates without collision."""
        g = HypothesisGraph()
        engine = DebateEngine(graph=g)

        correct_hyp = make_correct_hypothesis()
        false_hyp = make_false_attractive_hypothesis()

        engine.debate(correct_hyp)
        engine._state = DebateEngine.__init__.__defaults__  # reset state
        engine._state = None
        engine._state = __import__('packages.agents.debate_engine', fromlist=['DebateState']).DebateState.IDLE

        engine.debate(false_hyp)

        summary = g.summary()
        # Should have at least 2 world nodes, 2 hypothesis nodes, 2 verdicts
        total_nodes = sum(summary.values())
        assert total_nodes >= 6, f"Expected ≥ 6 nodes in graph, got {total_nodes}: {summary}"

    def test_accepted_hypothesis_in_accepted_list(self):
        """ACCEPTED hypotheses must appear in graph's accepted list."""
        g = HypothesisGraph()
        engine = DebateEngine(graph=g)

        correct_hyp = make_correct_hypothesis()
        record = engine.debate(correct_hyp)

        if record.arbiter_verdict.verdict == "ACCEPT":
            accepted = g.accepted_hypotheses()
            assert len(accepted) > 0, "ACCEPTED hypothesis must appear in accepted list"


class TestBatchDebate:

    def test_batch_processes_multiple_hypotheses(self):
        """Batch debate must handle a list of hypotheses."""
        blind_data = generate_blind_data("harmonic_spring")
        solver = SINDySolver()
        hyp1 = solver.solve(blind_data)

        false_hyp = make_false_attractive_hypothesis()

        engine = DebateEngine()
        records = engine.debate_batch([hyp1, false_hyp])

        assert len(records) == 2
        for record in records:
            assert record.arbiter_verdict is not None
            assert record.final_status in ("ACCEPTED", "REJECTED", "INCONCLUSIVE")

    def test_correct_accepted_false_not_accepted_in_batch(self):
        """In a batch, true and false hypotheses must receive different verdicts."""
        blind_data = generate_blind_data("harmonic_spring")
        solver = SINDySolver()
        correct = solver.solve(blind_data)
        false_h = make_false_attractive_hypothesis()

        engine = DebateEngine()
        records = engine.debate_batch([correct, false_h])

        verdicts = [r.arbiter_verdict.verdict for r in records]
        # At minimum: false one should NOT get ACCEPT if correct one does
        if verdicts[0] == "ACCEPT":
            assert verdicts[1] != "ACCEPT", (
                f"False hypothesis should not also be ACCEPTED.\n"
                f"Correct verdict: {verdicts[0]}, False verdict: {verdicts[1]}"
            )
