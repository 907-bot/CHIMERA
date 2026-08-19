"""Immutable Core Models for Continuum & Field Dynamics Engine (CHIMERA v2.0 - Phase 10)"""

from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Grid2D(BaseModel):
    """Specification of a 2D spatial discretization grid."""
    model_config = ConfigDict(frozen=True)

    nx: int = Field(gt=1, default=64, description="Number of grid points in x")
    ny: int = Field(gt=1, default=64, description="Number of grid points in y")
    lx: float = Field(gt=0.0, default=1.0, description="Domain physical length in x")
    ly: float = Field(gt=0.0, default=1.0, description="Domain physical length in y")

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny


class ContinuumConfig(BaseModel):
    """Configuration for Continuum and Field Dynamics World."""
    model_config = ConfigDict(frozen=True)

    world_id: str = "continuum_001"
    grid: Grid2D = Grid2D()
    dt: float = Field(gt=0.0, default=0.001)
    viscosity: float = Field(ge=0.0, default=0.01, description="Kinematic viscosity nu")
    thermal_diffusivity: float = Field(ge=0.0, default=0.01, description="Thermal conductivity alpha")
    c_light: float = Field(gt=0.0, default=1.0, description="Propagation speed for Maxwell equations")
    density: float = Field(gt=0.0, default=1.0, description="Fluid density rho")
    seed: int = 42
    boundary_condition: str = "periodic"  # 'periodic', 'dirichlet', 'neumann'


class FluidState(BaseModel):
    """Snapshot of a 2D Incompressible Fluid State."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    step: int = Field(ge=0, default=0)
    time: float = Field(ge=0.0, default=0.0)
    u_velocity: Tuple[Tuple[float, ...], ...]  # (ny, nx) x-velocity grid
    v_velocity: Tuple[Tuple[float, ...], ...]  # (ny, nx) y-velocity grid
    pressure: Tuple[Tuple[float, ...], ...]    # (ny, nx) scalar pressure field

    @classmethod
    def from_numpy(cls, step: int, time: float, u: np.ndarray, v: np.ndarray, p: np.ndarray) -> FluidState:
        return cls(
            step=step,
            time=time,
            u_velocity=tuple(tuple(float(x) for x in row) for row in u),
            v_velocity=tuple(tuple(float(x) for x in row) for row in v),
            pressure=tuple(tuple(float(x) for x in row) for row in p),
        )

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = np.array(self.u_velocity, dtype=np.float64)
        v = np.array(self.v_velocity, dtype=np.float64)
        p = np.array(self.pressure, dtype=np.float64)
        return u, v, p


class FieldState(BaseModel):
    """Snapshot of a 2D Continuous Scalar/Vector Field."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    step: int = Field(ge=0, default=0)
    time: float = Field(ge=0.0, default=0.0)
    scalar_field: Tuple[Tuple[float, ...], ...]  # Temperature or electric potential
    field_name: str = "temperature"

    @classmethod
    def from_numpy(cls, step: int, time: float, data: np.ndarray, field_name: str = "temperature") -> FieldState:
        return cls(
            step=step,
            time=time,
            scalar_field=tuple(tuple(float(x) for x in row) for row in data),
            field_name=field_name,
        )

    def to_numpy(self) -> np.ndarray:
        return np.array(self.scalar_field, dtype=np.float64)
