"""Immutable Core Models for Planetary & Cosmological Scale Engine (CHIMERA v5.0 - Phase 13)"""

from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any
from pydantic import BaseModel, ConfigDict, Field


class CelestialBody(BaseModel):
    """Immutable celestial body in cosmological / planetary orbit."""
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    mass: float = Field(gt=0.0, default=1.0)
    radius: float = Field(gt=0.0, default=1.0)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class CosmologicalWorldConfig(BaseModel):
    """Configuration for N-body cosmological and orbital system."""
    model_config = ConfigDict(frozen=True)

    world_id: str = "cosmo_001"
    g_grav: float = Field(gt=0.0, default=1.0, description="Universal gravitational constant")
    softening: float = Field(ge=0.0, default=0.01)
    dt: float = Field(gt=0.0, default=0.01)
    bodies: List[CelestialBody] = Field(default_factory=list)


class ClimateGridState(BaseModel):
    """Latitudinal planetary climate state."""
    model_config = ConfigDict(frozen=True)

    step: int = Field(ge=0, default=0)
    time: float = Field(ge=0.0, default=0.0)
    latitudes: Tuple[float, ...]  # -90 to +90 degrees
    temperatures: Tuple[float, ...]  # in Kelvin
    ice_coverage: Tuple[float, ...]  # fraction [0, 1]
    co2_ppm: float = 280.0
    solar_constant: float = 1361.0  # W/m^2


class MultiScaleState(BaseModel):
    """Hierarchical state coupling macroscopic planetary cosmos to microscopic local dynamics."""
    model_config = ConfigDict(frozen=True)

    cosmic_step: int = 0
    cosmic_time: float = 0.0
    bodies: List[CelestialBody]
    climate: ClimateGridState
    micro_substeps: int = 100
