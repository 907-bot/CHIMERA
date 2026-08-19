"""Immutable Core Models for Inter-Civilization Science & Meta-Theories (CHIMERA v7.0 - Phase 15)"""

from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any
from pydantic import BaseModel, ConfigDict, Field


class CrossUniverseMorphism(BaseModel):
    """Mathematical mapping translating physical variables and invariants from Universe A to Universe B."""
    model_config = ConfigDict(frozen=True)

    morphism_id: str
    source_universe_id: str
    target_universe_id: str
    variable_mappings: Dict[str, str]  # e.g., {"r": "d", "G": "k_gravity"}
    scaling_factors: Dict[str, float]  # e.g., {"time_dilation": 2.0, "mass_scale": 0.5}


class MetaInvariant(BaseModel):
    """A higher-order invariant holding true across multiple independent universe families."""
    model_config = ConfigDict(frozen=True)

    invariant_id: str
    name: str
    symbolic_form: str  # e.g., "E_total = const", "d/dt (Angular_Momentum) = 0"
    participating_universes: Tuple[str, ...]
    confidence_score: float = Field(ge=0.0, le=1.0, default=1.0)
    p_value: float = Field(ge=0.0, default=0.0)


class MetaTheoreticDAG(BaseModel):
    """Higher-order knowledge graph connecting local theories to overarching Meta-Theories of Everything (TOE)."""
    model_config = ConfigDict(frozen=True)

    graph_id: str
    nodes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    edges: List[Tuple[str, str, str]] = Field(default_factory=list)  # (source_id, target_id, relation_type)
    meta_invariants: List[MetaInvariant] = Field(default_factory=list)
