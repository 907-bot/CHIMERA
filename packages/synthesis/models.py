"""Immutable Core Models for World-Engineering & Causal Universe Synthesis (CHIMERA v8.0 - Phase 16)"""

from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any
from pydantic import BaseModel, ConfigDict, Field


class LawKernel(BaseModel):
    """Parametric force law / interaction kernel definition."""
    model_config = ConfigDict(frozen=True)

    kernel_name: str
    gravity_exponent: float = 2.0  # F ∝ 1/r^p
    repulsion_exponent: float = 12.0  # Lennard-Jones style repulsion
    coupling_strength: float = 1.0
    crossover_radius: float = 1.0


class SynthesisTarget(BaseModel):
    """Target emergent metric desired by Universe Architect."""
    model_config = ConfigDict(frozen=True)

    target_metric: str = "cluster_diversity"  # 'cluster_diversity', 'orbit_stability', 'reaction_cycles'
    target_value: float = 10.0
    tolerance: float = 0.5


class UniverseSpecification(BaseModel):
    """Complete specification of a synthetic universe designed by AI Architect."""
    model_config = ConfigDict(frozen=True)

    spec_id: str
    law_kernel: LawKernel
    target_objectives: List[SynthesisTarget]
    fitness_achieved: float = 0.0
