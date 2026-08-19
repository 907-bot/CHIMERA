"""Immutable Core Models for Macro-Biochemistry & Abiogenesis Engine (CHIMERA v4.0 - Phase 12)"""

from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any
from pydantic import BaseModel, ConfigDict, Field


class Monomer(BaseModel):
    """Monomer unit in coarse-grained 3D molecular chain (H: Hydrophobic, P: Polar, C: Catalytic)."""
    model_config = ConfigDict(frozen=True)

    id: int
    monomer_type: str = "H"  # 'H', 'P', 'C' (Catalytic)
    position: Tuple[int, int, int] = (0, 0, 0)


class Polymer3D(BaseModel):
    """3D coarse-grained folded polymer chain."""
    model_config = ConfigDict(frozen=True)

    id: str
    sequence: str = "HPHPPHHPH"
    coordinates: Tuple[Tuple[int, int, int], ...]
    energy: float = 0.0
    is_catalytic: bool = False


class VesicleMembrane(BaseModel):
    """Vesicular lipid compartment enclosing a metabolic interior."""
    model_config = ConfigDict(frozen=True)

    vesicle_id: str
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = Field(gt=0.0, default=5.0)
    lipid_count: int = Field(gt=0, default=100)
    internal_metabolites: Dict[str, float] = Field(default_factory=dict)
    permeability: float = Field(ge=0.0, le=1.0, default=0.2)


class HypercycleState(BaseModel):
    """Concentrations of an autocatalytic hypercycle network."""
    model_config = ConfigDict(frozen=True)

    step: int = Field(ge=0, default=0)
    time: float = Field(ge=0.0, default=0.0)
    species_concentrations: Tuple[float, ...]
    species_names: Tuple[str, ...]
