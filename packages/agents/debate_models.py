"""Structured Data Models for the CHIMERA Adversarial Scientific Society.

All inter-agent communication uses these Pydantic schemas — no free-text.
This enforces single-pass structured JSON rounds per the cost-optimization rule.

Hierarchy:
  BullArgument       : Strongest evidence supporting the hypothesis
  BearArgument       : Alternative explanations and identified weaknesses
  CounterfactualExperiment : Skeptic's proposed falsification experiment
  ExperimentResult   : Outcome of running the counterfactual
  ArbiterVerdict     : Final scored decision from Arbiter
  DebateRecord       : Complete immutable record of one debate session
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Bull Agent Output
# ---------------------------------------------------------------------------

class SupportingEvidence(BaseModel):
    """A single piece of quantitative evidence cited by the Bull agent."""
    model_config = ConfigDict(frozen=True)

    evidence_type: Literal["r_squared", "rmse", "prediction", "conservation", "symmetry", "convergence"]
    """Category of the evidence."""

    value: float
    """Numerical value of the evidence metric."""

    description: str
    """Human-readable explanation of why this supports the hypothesis."""

    step_range: Optional[tuple] = None
    """(start, end) steps from trajectory where evidence was observed."""


class BullArgument(BaseModel):
    """Bull agent: presents the strongest quantitative case FOR the hypothesis.

    Strategy: cite best-performing metrics, high R², convergence across seeds,
    conservation law compliance, and dimensional consistency.
    """
    model_config = ConfigDict(frozen=True)

    agent: str = "Bull"
    hypothesis_id: str
    world_name: str

    confidence_score: float = Field(ge=0.0, le=1.0)
    """Overall confidence in the hypothesis being correct [0, 1]."""

    supporting_evidence: List[SupportingEvidence]
    """List of quantitative evidence items supporting the hypothesis."""

    strongest_claim: str
    """The single most compelling argument in favour of the hypothesis."""

    predicted_generalisation: str
    """How the Bull agent claims this law will generalise to new worlds."""

    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Bear Agent Output
# ---------------------------------------------------------------------------

class Weakness(BaseModel):
    """A specific weakness or alternative explanation identified by the Bear agent."""
    model_config = ConfigDict(frozen=True)

    weakness_type: Literal[
        "overfitting", "confounder", "symmetry_break",
        "limited_data", "alternative_law", "numerical_artifact"
    ]
    description: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    suggested_test: Optional[str] = None
    """Optional: what experiment would distinguish this weakness."""


class BearArgument(BaseModel):
    """Bear agent: identifies the strongest weaknesses AGAINST the hypothesis.

    Strategy: look for overfitting, alternative equally valid equations,
    data range limitations, and potential confounders.
    """
    model_config = ConfigDict(frozen=True)

    agent: str = "Bear"
    hypothesis_id: str
    world_name: str

    doubt_score: float = Field(ge=0.0, le=1.0)
    """Overall doubt score — probability hypothesis is WRONG [0, 1]."""

    weaknesses: List[Weakness]
    """Identified weaknesses ordered by severity (highest first)."""

    alternative_hypothesis: Optional[str] = None
    """If Bear proposes a competing equation, it goes here."""

    critical_flaw: str
    """The single most damaging argument against the hypothesis."""

    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Skeptic Agent Output
# ---------------------------------------------------------------------------

class PerturbationSpec(BaseModel):
    """Specification for a single perturbation to apply in a counterfactual world."""
    model_config = ConfigDict(frozen=True)

    parameter: str
    """What is being perturbed, e.g. 'initial_displacement', 'seed', 'amplitude'."""

    original_value: float
    perturbed_value: float

    rationale: str
    """Why this perturbation would falsify the hypothesis if the law is wrong."""


class CounterfactualExperiment(BaseModel):
    """Skeptic's proposed experiment designed to break the hypothesis.

    The experiment must be concrete and runnable by the Intervention Engine.
    If the hypothesis truly holds, it must survive the perturbation.
    If the hypothesis is fragile (overfitted / wrong), it will collapse.
    """
    model_config = ConfigDict(frozen=True)

    agent: str = "Skeptic"
    hypothesis_id: str
    world_name: str

    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    experiment_name: str
    description: str

    perturbations: List[PerturbationSpec]
    """List of changes to apply to generate the counterfactual world."""

    predicted_outcome_if_true: str
    """What the Skeptic predicts will happen if the hypothesis is correct."""

    predicted_outcome_if_false: str
    """What the Skeptic predicts will happen if the hypothesis is wrong."""

    r2_threshold_to_survive: float = Field(default=0.95, ge=0.0, le=1.0)
    """Minimum R² the hypothesis must achieve on perturbed world to survive."""

    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Experiment Result (Intervention Engine output)
# ---------------------------------------------------------------------------

class ExperimentResult(BaseModel):
    """Outcome of running a CounterfactualExperiment through the Intervention Engine."""
    model_config = ConfigDict(frozen=True)

    experiment_id: str
    hypothesis_id: str
    world_name: str

    r_squared_on_perturbed: float
    """R² of hypothesis prediction on the perturbed world trajectory."""

    rmse_on_perturbed: float

    survived: bool
    """True if r_squared_on_perturbed ≥ experiment.r2_threshold_to_survive."""

    interpretation: str
    """System-generated explanation of what the result means for the hypothesis."""

    run_duration_seconds: float
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Arbiter Verdict
# ---------------------------------------------------------------------------

class EvidenceWeight(BaseModel):
    """Arbiter's weighting of a single piece of evidence."""
    model_config = ConfigDict(frozen=True)

    source: Literal["Bull", "Bear", "Skeptic", "Experiment"]
    weight: float = Field(ge=0.0, le=1.0)
    contribution: str


class ArbiterVerdict(BaseModel):
    """Arbiter's final quantitative decision after reviewing all arguments.

    Arbiter is purely computational — no LLM inference.
    It weighs Bull/Bear arguments and Experiment results using Bayesian scoring.
    """
    model_config = ConfigDict(frozen=True)

    agent: str = "Arbiter"
    hypothesis_id: str
    world_name: str

    verdict: Literal["ACCEPT", "REJECT", "INCONCLUSIVE"]
    """Final disposition of the hypothesis."""

    bayesian_confidence: float = Field(ge=0.0, le=1.0)
    """Posterior probability the hypothesis is correct given all evidence."""

    evidence_weights: List[EvidenceWeight]
    """How each source of evidence was weighted."""

    reasoning: str
    """Arbiter's final structured explanation of the verdict."""

    reproducibility_score: float = Field(ge=0.0, le=1.0)
    """Fraction of counterfactual experiments the hypothesis survived."""

    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Complete Debate Record
# ---------------------------------------------------------------------------

class DebateRecord(BaseModel):
    """Complete immutable record of one adversarial debate session.

    Per AGENTS.md Rule 6: All debate records, including failed hypotheses,
    are retained permanently as scientific evidence.
    """
    model_config = ConfigDict(frozen=True)

    debate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis_id: str
    world_name: str

    bull_argument: BullArgument
    bear_argument: BearArgument
    skeptic_experiment: CounterfactualExperiment
    experiment_result: ExperimentResult
    arbiter_verdict: ArbiterVerdict

    final_status: Literal["ACCEPTED", "REJECTED", "INCONCLUSIVE"]
    """Mirrors arbiter_verdict.verdict for quick lookup."""

    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    duration_seconds: float = 0.0
