"""Data Models for CHIMERA Scientific Civilization & In-World Observers (Phases 9 & 10).

Defines schemas for:
  - InWorldObserver (observer entity living inside the simulation)
  - CivilizationExperiment (experiment conducted by an in-world observer)
  - CivilizationTheory (theory formulated and archived by in-world scientists)
  - ScientificCivilizationState (cumulative scientific knowledge of the civilization)
  - CivilizationSimulationResult (complete simulation of a scientific civilization)
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, ConfigDict, Field
from packages.core.models import Vector2D


class InWorldObserver(BaseModel):
    """An observer agent existing inside the simulation that conducts experiments."""
    model_config = ConfigDict(frozen=False)

    observer_id: str = Field(default_factory=lambda: f"obs_{uuid.uuid4().hex[:6]}")
    name: str = "Observer_1"
    specialization: Literal["PHYSICS", "CHEMISTRY", "ECOLOGY"] = "PHYSICS"
    position: Vector2D = Field(default_factory=lambda: Vector2D(x=50.0, y=50.0))
    measurement_noise_std: float = Field(default=0.01, ge=0.0, description="Instrument measurement error")
    theories_formulated: int = 0
    theories_accepted: int = 0


class CivilizationExperiment(BaseModel):
    """An experiment designed and executed by an in-world observer."""
    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:6]}")
    observer_id: str
    target_phenomenon: str
    intervention_type: str
    sample_size: int
    measured_r_squared: float
    conclusion: str


class CivilizationTheory(BaseModel):
    """A scientific law / theorem formulated and certified by in-world observers."""
    model_config = ConfigDict(frozen=True)

    theory_id: str = Field(default_factory=lambda: f"th_{uuid.uuid4().hex[:6]}")
    author_observer_id: str
    title: str
    mathematical_formula: str
    domain: Literal["PHYSICS", "CHEMISTRY", "ECOLOGY"]
    evidence_experiments: List[str] = Field(default_factory=list)
    consensus_score: float = Field(ge=0.0, le=1.0, description="Fraction of in-world observer peer votes in favor")
    status: Literal["PEER_REVIEW", "ACCEPTED_PARADIGM", "FALSIFIED_THEORY"] = "PEER_REVIEW"
    created_generation: int = 0


class ScientificCivilizationState(BaseModel):
    """The collective state of an in-world scientific civilization."""
    model_config = ConfigDict(frozen=True)

    generation: int
    active_observers: int
    total_experiments_conducted: int
    accepted_theories_count: int
    falsified_theories_count: int
    scientific_consensus_index: float = Field(ge=0.0, le=1.0)


class CivilizationSimulationResult(BaseModel):
    """Complete history of an evolving scientific civilization."""
    model_config = ConfigDict(frozen=True)

    civilization_id: str = Field(default_factory=lambda: f"civ_{uuid.uuid4().hex[:8]}")
    total_generations: int
    observers: List[InWorldObserver]
    archived_theories: List[CivilizationTheory]
    experiments_log: List[CivilizationExperiment]
    timeline_snapshots: List[ScientificCivilizationState]
    paradigm_count: int
    accuracy_vs_ground_truth: float = Field(
        ge=0.0, le=1.0,
        description="Meta-metric: percentage of accepted in-world theories matching true engine laws"
    )
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
