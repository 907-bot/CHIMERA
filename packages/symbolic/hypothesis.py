"""Hypothesis Data Model for CHIMERA Phase 3 Symbolic Discovery Engine.

A Hypothesis is the formal structured output of the symbolic solvers:
  - A candidate equation (SymPy expression string)
  - Fitted parameter values
  - Goodness-of-fit metrics (R², RMSE)
  - Evidence citations (trajectory step range used for fitting)
  - Status lifecycle: CANDIDATE → VALIDATED | FALSIFIED

Per AGENTS.md Rule 6: Failed hypotheses MUST be retained in scientific records.
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class HypothesisParameters(BaseModel):
    """Container for fitted parameter values with uncertainty estimates."""
    model_config = ConfigDict(frozen=True)

    values: Dict[str, float]
    """Map of parameter name → fitted value."""

    uncertainties: Dict[str, float] = Field(default_factory=dict)
    """Map of parameter name → 1-sigma uncertainty (if available)."""


class PredictionMetrics(BaseModel):
    """Quantitative performance metrics for a hypothesis prediction."""
    model_config = ConfigDict(frozen=True)

    r_squared: float = Field(ge=-1.0, le=1.0, description="Coefficient of determination R²")
    rmse: float = Field(ge=0.0, description="Root Mean Squared Error on test trajectory")
    mae: float = Field(ge=0.0, description="Mean Absolute Error on test trajectory")
    train_steps: int = Field(ge=0, description="Number of steps used for fitting")
    test_steps: int = Field(ge=0, description="Number of held-out steps used for validation")


class Hypothesis(BaseModel):
    """Formal scientific hypothesis produced by the symbolic discovery engine.

    A Hypothesis links a candidate equation to observable evidence, quantified
    goodness-of-fit, and a lifecycle status. Per CHIMERA governance:
    - CANDIDATE: equation proposed by solver, not yet validated
    - VALIDATED:  prediction R² > 0.99 on held-out test trajectory
    - FALSIFIED:  prediction fails or counter-evidence found (IMMUTABLE — never deleted)
    """
    model_config = ConfigDict(frozen=False)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Unique hypothesis UUID."""

    world_name: str
    """Name of the benchmark world this hypothesis was derived from."""

    solver: str
    """Solver that generated this hypothesis, e.g. 'SINDy', 'PySR', 'manual'."""

    candidate_equation: str
    """SymPy-compatible expression string, e.g. '-2.99*x'."""

    parameters: HypothesisParameters
    """Fitted parameter values and uncertainties."""

    metrics: Optional[PredictionMetrics] = None
    """Filled in after validation against held-out trajectory."""

    evidence_step_range: tuple[int, int] = (0, 800)
    """(start_step, end_step) used as training window."""

    status: Literal["CANDIDATE", "VALIDATED", "FALSIFIED"] = "CANDIDATE"
    """Lifecycle status. FALSIFIED hypotheses are retained permanently."""

    falsification_evidence: Optional[str] = None
    """If FALSIFIED, human/system-readable explanation of counter-evidence."""

    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def with_metrics(self, metrics: PredictionMetrics) -> Hypothesis:
        """Return a new Hypothesis with updated prediction metrics."""
        return self.model_copy(update={"metrics": metrics})

    def validate(self, metrics: PredictionMetrics, threshold: float = 0.99) -> Hypothesis:
        """Score hypothesis against held-out trajectory and set VALIDATED/FALSIFIED.

        Args:
            metrics:   Computed prediction metrics on held-out trajectory.
            threshold: R² threshold above which hypothesis is VALIDATED.

        Returns:
            New Hypothesis with updated status and metrics.
        """
        new_status: Literal["CANDIDATE", "VALIDATED", "FALSIFIED"]
        evidence: Optional[str] = None

        if metrics.r_squared >= threshold:
            new_status = "VALIDATED"
        else:
            new_status = "FALSIFIED"
            evidence = (
                f"R²={metrics.r_squared:.4f} < threshold={threshold:.4f}. "
                f"RMSE={metrics.rmse:.6f}. Prediction failed on "
                f"{metrics.test_steps} held-out steps."
            )

        return self.model_copy(update={
            "metrics": metrics,
            "status": new_status,
            "falsification_evidence": evidence,
        })

    def summary(self) -> str:
        """Return a compact human-readable summary string."""
        r2 = self.metrics.r_squared if self.metrics else float("nan")
        return (
            f"[{self.status}] {self.solver} | {self.world_name} | "
            f"Eq: {self.candidate_equation} | R²={r2:.4f}"
        )
