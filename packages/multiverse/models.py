"""Data Models for CHIMERA Multiverse Engine & Cross-World Discovery.

Defines schemas for:
  - WorldFamilyType (Family A, B, C, D)
  - WorldFamilySpec (Configuration for generating parallel worlds)
  - WorldBranchSpec (Specification for checkpoint branching in Family D)
  - LyapunovResult (Chaos metrics & exponent estimation)
  - InvariantResult (Statistical validation of cross-world invariants)
  - MultiverseBatchResult (Container for batch simulation runs)
"""

from __future__ import annotations
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, ConfigDict, Field
from packages.core.models import WorldConfig, WorldState, Vector2D


class WorldFamilyType(str, Enum):
    """Classification of Multiverse World Families."""
    FAMILY_A_INITIAL_CONDITIONS = "family_a_initial_conditions"
    """Identical physics laws, varying random seeds / initial conditions."""

    FAMILY_B_CHAOS_LYAPUNOV = "family_b_chaos_lyapunov"
    """Twin worlds with micro-perturbations (1e-8) to measure Lyapunov exponents."""

    FAMILY_C_PARAMETER_SWEEP = "family_c_parameter_sweep"
    """Systematic sweeps over physical constants (G, mass, restitution, dt)."""

    FAMILY_D_BRANCHING_CHECKPOINT = "family_d_branching_checkpoint"
    """Branching worlds from a parent checkpoint at step k with intervention."""


class WorldBranchSpec(BaseModel):
    """Specification for branching a child world from a parent checkpoint."""
    model_config = ConfigDict(frozen=True)

    parent_world_id: str
    branch_step: int = Field(ge=0, description="Step at which to fork the child world")
    child_world_id: str = Field(default_factory=lambda: f"branch_{uuid.uuid4().hex[:8]}")
    
    # Interventions to apply at branch point
    velocity_perturbations: Dict[int, Vector2D] = Field(
        default_factory=dict,
        description="Map of particle_id -> added delta velocity Vector2D"
    )
    mass_perturbations: Dict[int, float] = Field(
        default_factory=dict,
        description="Map of particle_id -> new mass value"
    )
    new_gravity: Optional[float] = None
    description: str = "Counterfactual branching intervention"


class WorldFamilySpec(BaseModel):
    """Specification for executing a Multiverse World Family."""
    model_config = ConfigDict(frozen=True)

    family_id: str = Field(default_factory=lambda: f"fam_{uuid.uuid4().hex[:8]}")
    family_type: WorldFamilyType
    base_config: WorldConfig
    num_worlds: int = Field(default=10, ge=1, description="Number of parallel worlds to generate")
    steps_per_world: int = Field(default=200, ge=1, description="Simulation steps per world")
    
    # Family-specific parameters
    parameter_sweep_key: Optional[str] = None
    """For Family C: Config field name to sweep (e.g. 'gravity_constant', 'restitution')."""

    parameter_sweep_values: Optional[List[float]] = None
    """For Family C: Explicit list of values to assign across worlds."""

    chaos_perturbation_epsilon: float = Field(
        default=1e-8,
        description="For Family B: Position/velocity micro-perturbation magnitude"
    )

    branch_specs: Optional[List[WorldBranchSpec]] = None
    """For Family D: Explicit branch specifications."""


class LyapunovResult(BaseModel):
    """Quantitative chaos analysis result from twin trajectory divergence."""
    model_config = ConfigDict(frozen=True)

    base_world_id: str
    perturbed_world_id: str
    epsilon: float
    lyapunov_exponent: float = Field(description="Max Lyapunov exponent lambda (estimated via linear fit of ln(delta/delta0))")
    is_chaotic: bool = Field(description="True if lambda > 0 with statistical significance")
    r_squared_fit: float = Field(description="R^2 of the exponential divergence log-linear fit")
    divergence_history: List[float] = Field(description="Phase-space Euclidean distance delta(t) at each recorded step")
    time_steps: List[float] = Field(description="Corresponding simulation time t")
    classification: Literal["CHAOTIC", "REGULAR_PERIODIC", "NEUTRAL_DAMPED"] = "REGULAR_PERIODIC"


class InvariantCandidate(str, Enum):
    """Candidate physical quantities evaluated for invariance."""
    TOTAL_ENERGY = "total_energy"
    TOTAL_MOMENTUM_X = "total_momentum_x"
    TOTAL_MOMENTUM_Y = "total_momentum_y"
    CENTER_OF_MASS_VELOCITY = "center_of_mass_velocity"
    PARTICLE_POSITION_SAMPLE = "particle_position_sample"
    PARTICLE_VELOCITY_SAMPLE = "particle_velocity_sample"


class InvariantResult(BaseModel):
    """Statistical evaluation of whether a quantity is a universal physical invariant."""
    model_config = ConfigDict(frozen=True)

    quantity_name: str
    is_universal_invariant: bool = Field(
        description="True if quantity is conserved within worlds AND holds universally across all seeds"
    )
    mean_within_world_drift: float = Field(
        description="Mean relative change |Q(t) - Q(0)| / |Q(0)| across all worlds"
    )
    max_within_world_drift: float = Field(
        description="Max relative drift across any single world"
    )
    across_world_variance: float = Field(
        description="Variance of the quantity across different initial seeds"
    )
    conservation_score: float = Field(
        ge=0.0, le=1.0,
        description="Normalized metric [0, 1] where 1.0 is exact bitwise conservation"
    )
    verdict: Literal["UNIVERSAL_CONSERVATION_LAW", "SEED_CONTINGENT_HISTORICAL_FACT", "DISSIPATIVE_ASYMMETRY"]
    description: str


class MultiverseBatchResult(BaseModel):
    """Result container for a completed Multiverse Family execution."""
    model_config = ConfigDict(frozen=True)

    family_id: str
    family_type: WorldFamilyType
    total_worlds: int
    steps_per_world: int
    world_ids: List[str]
    elapsed_seconds: float
    invariants: Optional[List[InvariantResult]] = None
    lyapunov_summary: Optional[LyapunovResult] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
