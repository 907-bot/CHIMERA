"""Immutable Core Data Models for Quantum & Statistical Mechanics Multiverse (CHIMERA v3.0 - Phase 11)"""

from __future__ import annotations
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class LatticeHamiltonianConfig(BaseModel):
    """Specification of 1D/2D Discrete Quantum Lattice Hamiltonian."""
    model_config = ConfigDict(frozen=True)

    lattice_size: int = Field(gt=2, default=32, description="Number of discrete spatial lattice sites")
    dx: float = Field(gt=0.0, default=0.1, description="Spatial lattice step")
    hbar: float = Field(gt=0.0, default=1.0, description="Reduced Planck constant")
    mass: float = Field(gt=0.0, default=1.0, description="Particle effective mass")
    dt: float = Field(gt=0.0, default=0.005, description="Time evolution step")
    potential_type: str = "harmonic"  # 'harmonic', 'barrier', 'double_well', 'free'
    barrier_height: float = 10.0
    barrier_width: float = 1.0


class QuantumLatticeState(BaseModel):
    """Immutable state vector on discrete quantum lattice."""
    model_config = ConfigDict(frozen=True)

    step: int = Field(ge=0, default=0)
    time: float = Field(ge=0.0, default=0.0)
    branch_id: str = "root"
    psi_real: Tuple[float, ...]
    psi_imag: Tuple[float, ...]

    @classmethod
    def from_complex_array(cls, psi: np.ndarray, step: int = 0, time: float = 0.0, branch_id: str = "root") -> QuantumLatticeState:
        return cls(
            step=step,
            time=time,
            branch_id=branch_id,
            psi_real=tuple(float(x) for x in np.real(psi)),
            psi_imag=tuple(float(x) for x in np.imag(psi)),
        )

    def to_complex_array(self) -> np.ndarray:
        return np.array(self.psi_real, dtype=np.float64) + 1j * np.array(self.psi_imag, dtype=np.float64)

    @property
    def probability_density(self) -> np.ndarray:
        psi = self.to_complex_array()
        return np.abs(psi) ** 2

    @property
    def total_probability(self) -> float:
        return float(np.sum(self.probability_density))


class BranchNode(BaseModel):
    """Many-Worlds Decoherence Tree Node representing an observation/measurement split."""
    model_config = ConfigDict(frozen=True)

    branch_id: str
    parent_branch_id: Optional[str] = None
    step_created: int = 0
    measurement_outcome: str = ""
    branch_probability: float = 1.0
    state_vector: QuantumLatticeState
