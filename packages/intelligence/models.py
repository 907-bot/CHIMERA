"""Data Models for CHIMERA Embodied Intelligence & Emergence (Phase 8).

Defines schemas for:
  - NeuralPolicy (weights for sensory-motor mapping)
  - SensoryObservation (food vector, rival vector, energy level, signal)
  - AgentAction (velocity displacement, signal emission, sharing)
  - InformationMetrics (Transfer Entropy, Mutual Information, Polarization)
  - SocialSimulationResult (multi-agent trajectory with emergence metrics)
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from packages.core.models import Vector2D


class NeuralPolicy(BaseModel):
    """Compact sensory-motor neural network policy parameters."""
    model_config = ConfigDict(frozen=True)

    weights: List[List[float]] = Field(
        description="Weight matrix of shape (input_dim=5, output_dim=3)"
    )
    bias: List[float] = Field(
        description="Bias vector of length 3"
    )

    @classmethod
    def create_random(cls, rng: np.random.Generator) -> NeuralPolicy:
        """Create a randomized neural policy."""
        w = rng.normal(0.0, 0.5, size=(5, 3)).tolist()
        b = rng.normal(0.0, 0.1, size=(3,)).tolist()
        return cls(weights=w, bias=b)


class SensoryObservation(BaseModel):
    """Sensory inputs perceived by an agent at step t."""
    model_config = ConfigDict(frozen=True)

    food_dx: float
    food_dy: float
    nearest_agent_dx: float
    nearest_agent_dy: float
    current_energy: float

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.food_dx / 50.0,
            self.food_dy / 50.0,
            self.nearest_agent_dx / 50.0,
            self.nearest_agent_dy / 50.0,
            self.current_energy / 50.0,
        ], dtype=np.float64)


class AgentAction(BaseModel):
    """Motor outputs executed by an agent."""
    model_config = ConfigDict(frozen=True)

    move_dx: float
    move_dy: float
    broadcast_signal: float


class InformationMetrics(BaseModel):
    """Information-theoretic metrics quantifying emergent social coordination."""
    model_config = ConfigDict(frozen=True)

    transfer_entropy: float = Field(
        ge=0.0,
        description="Transfer Entropy T_Y->X measuring directed information flow between agents"
    )
    mutual_information: float = Field(
        ge=0.0,
        description="Mutual Information I(X; Y) between agent spatial orientations"
    )
    swarm_polarization: float = Field(
        ge=0.0, le=1.0,
        description="Global order parameter Phi = (1/N) * ||sum v_i / ||v_i||||"
    )
    is_collective_emergence: bool = Field(
        description="True if transfer entropy and swarm order exceed emergence thresholds"
    )
    classification: Literal["COLLECTIVE_COORDINATION", "INDEPENDENT_AGENTS", "NOISE_DISPERSED"]
    description: str


class SocialSimulationResult(BaseModel):
    """Complete trajectory and emergence report for an embodied multi-agent simulation."""
    model_config = ConfigDict(frozen=True)

    simulation_id: str = Field(default_factory=lambda: f"soc_{uuid.uuid4().hex[:8]}")
    total_steps: int
    num_agents: int
    information_metrics: InformationMetrics
    mean_energy_history: List[float]
    polarization_history: List[float]
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
