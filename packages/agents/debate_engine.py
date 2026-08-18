"""Adversarial Debate Engine State Machine for CHIMERA Phase 4.

Orchestrates the full Bull → Bear → Skeptic → Intervention → Arbiter pipeline
as a strict single-pass state machine with no open-ended loops.

State Machine:
  IDLE
    │ debate(hypothesis)
    ▼
  BULL_ARGUING  → BullArgument
    │
  BEAR_ARGUING  → BearArgument
    │
  SKEPTIC_DESIGNING → CounterfactualExperiment
    │
  INTERVENING   → ExperimentResult
    │
  ARBITER_DECIDING → ArbiterVerdict
    │
  COMPLETE      → DebateRecord (immutable, appended to graph)

Per AGENTS.md: Antigravity manages state machine flow.
No LLM is invoked inside this module — roles.py handles agent logic.
"""

from __future__ import annotations
import time
from typing import Optional, List
from enum import Enum, auto

from packages.symbolic.hypothesis import Hypothesis
from packages.agents.debate_models import (
    BullArgument,
    BearArgument,
    CounterfactualExperiment,
    ExperimentResult,
    ArbiterVerdict,
    DebateRecord,
)
from packages.agents.roles import BullAgent, BearAgent, SkepticAgent, ArbiterAgent
from packages.agents.intervention import InterventionEngine
from packages.agents.hypothesis_graph import HypothesisGraph


class DebateState(Enum):
    """Lifecycle states of the adversarial debate state machine."""
    IDLE = auto()
    BULL_ARGUING = auto()
    BEAR_ARGUING = auto()
    SKEPTIC_DESIGNING = auto()
    INTERVENING = auto()
    ARBITER_DECIDING = auto()
    COMPLETE = auto()
    ERROR = auto()


class DebateEngine:
    """Single-pass adversarial debate state machine.

    Runs Bull → Bear → Skeptic → Intervention → Arbiter in strict sequence.
    Each agent produces exactly ONE structured JSON output — no loops.

    Args:
        graph:              Optional HypothesisGraph for provenance tracking.
        run_all_experiments: If True, runs the full battery of 3 experiments
                             and uses the mean survival rate for Arbiter input.
                             If False (default), runs only Skeptic's hardest test.
    """

    def __init__(
        self,
        graph: Optional[HypothesisGraph] = None,
        run_all_experiments: bool = False,
    ):
        self.bull = BullAgent()
        self.bear = BearAgent()
        self.skeptic = SkepticAgent()
        self.arbiter = ArbiterAgent()
        self.intervention = InterventionEngine()
        self.graph = graph
        self.run_all_experiments = run_all_experiments

        self._state = DebateState.IDLE
        self._current_debate_id: Optional[str] = None

    @property
    def state(self) -> DebateState:
        return self._state

    def debate(self, hypothesis: Hypothesis) -> DebateRecord:
        """Run the full adversarial debate pipeline for a single hypothesis.

        State transitions:
          IDLE → BULL_ARGUING → BEAR_ARGUING → SKEPTIC_DESIGNING
          → INTERVENING → ARBITER_DECIDING → COMPLETE

        Args:
            hypothesis: The Hypothesis to debate (must have metrics set).

        Returns:
            DebateRecord with all arguments, experiment, and verdict.

        Raises:
            ValueError: If hypothesis has no metrics (not yet validated).
        """
        if hypothesis.metrics is None:
            raise ValueError(
                f"Hypothesis {hypothesis.id[:8]}... has no metrics. "
                "Run SINDy validation before debating."
            )

        t_start = time.perf_counter()
        self._state = DebateState.BULL_ARGUING

        # --- Round 1: Bull argues FOR ---
        bull_arg: BullArgument = self.bull.argue(hypothesis)

        # --- Round 2: Bear argues AGAINST ---
        self._state = DebateState.BEAR_ARGUING
        bear_arg: BearArgument = self.bear.argue(hypothesis)

        # --- Round 3: Skeptic designs falsification experiment ---
        self._state = DebateState.SKEPTIC_DESIGNING
        skeptic_exp: CounterfactualExperiment = self.skeptic.design_experiment(hypothesis)

        # --- Round 4: Intervention Engine runs experiment ---
        self._state = DebateState.INTERVENING

        if self.run_all_experiments:
            all_experiments = self.intervention.design_standard_experiments(hypothesis)
            results: List[ExperimentResult] = [
                self.intervention.run_experiment(hypothesis, exp)
                for exp in all_experiments
            ]
            # Use the result with the lowest R² as the hardest test
            worst_result = min(results, key=lambda r: r.r_squared_on_perturbed)
            primary_result = worst_result
            primary_exp = next(
                e for e in all_experiments
                if e.experiment_id == worst_result.experiment_id
            )
        else:
            primary_result: ExperimentResult = self.intervention.run_experiment(
                hypothesis, skeptic_exp
            )
            primary_exp = skeptic_exp

        # --- Round 5: Arbiter issues verdict ---
        self._state = DebateState.ARBITER_DECIDING
        verdict: ArbiterVerdict = self.arbiter.issue_verdict(
            hypothesis, bull_arg, bear_arg, primary_exp, primary_result
        )

        # --- Build immutable DebateRecord ---
        elapsed = time.perf_counter() - t_start
        final_status_map = {"ACCEPT": "ACCEPTED", "REJECT": "REJECTED", "INCONCLUSIVE": "INCONCLUSIVE"}

        record = DebateRecord(
            hypothesis_id=hypothesis.id,
            world_name=hypothesis.world_name,
            bull_argument=bull_arg,
            bear_argument=bear_arg,
            skeptic_experiment=primary_exp,
            experiment_result=primary_result,
            arbiter_verdict=verdict,
            final_status=final_status_map[verdict.verdict],
            duration_seconds=round(elapsed, 4),
        )

        # --- Register in provenance graph (append-only) ---
        if self.graph is not None:
            self.graph.record_full_debate(record)

        self._state = DebateState.COMPLETE
        return record

    def debate_batch(self, hypotheses: List[Hypothesis]) -> List[DebateRecord]:
        """Run debates for a list of hypotheses sequentially.

        Args:
            hypotheses: List of Hypothesis objects to debate.

        Returns:
            List of DebateRecord objects in the same order.
        """
        records = []
        for hyp in hypotheses:
            self._state = DebateState.IDLE
            try:
                record = self.debate(hyp)
                records.append(record)
            except Exception as e:
                self._state = DebateState.ERROR
                raise RuntimeError(f"Debate failed for hypothesis {hyp.id[:8]}: {e}") from e
        return records
