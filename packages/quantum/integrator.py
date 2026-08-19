"""Unitary Quantum Lattice Integrator (CHIMERA v3.0 - Phase 11)

Solves discrete time-dependent Schrödinger equation on a lattice:
    i ℏ ∂ψ/∂t = Ĥ ψ = (- (ℏ² / 2m) ∇² + V(x)) ψ
Using exact unitary spectral propagator:
    Ĥ = V Λ V†
    U(dt) = V exp(-i Λ dt / ℏ) V†
    ψ(t + dt) = U(dt) ψ(t)
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from packages.quantum.models import LatticeHamiltonianConfig, QuantumLatticeState


class QuantumLatticeIntegrator:
    """Deterministic unitary Schrödinger time-evolution on 1D discrete lattice."""

    def __init__(self, config: LatticeHamiltonianConfig):
        self.config = config
        self.N = config.lattice_size
        self.dx = config.dx
        self.hbar = config.hbar
        self.m = config.mass
        self.dt = config.dt
        self.x = (np.arange(self.N) - self.N / 2) * self.dx

        # Construct kinetic energy operator T = - (ℏ² / 2m) d²/dx²
        diag_T = 2.0 * np.ones(self.N)
        off_diag_T = -1.0 * np.ones(self.N - 1)
        T_matrix = (self.hbar ** 2 / (2.0 * self.m * (self.dx ** 2))) * (
            np.diag(diag_T) + np.diag(off_diag_T, k=1) + np.diag(off_diag_T, k=-1)
        )

        # Construct potential energy matrix V
        V_diag = self._build_potential(config.potential_type)
        V_matrix = np.diag(V_diag)

        # Full Hermitian Hamiltonian Ĥ = T + V
        self.H = T_matrix + V_matrix

        # Exact unitary propagator via Hermitian eigendecomposition
        eigvals, eigvecs = eigh(self.H)
        diag_phase = np.exp(-1j * eigvals * self.dt / self.hbar)
        self.U = eigvecs @ np.diag(diag_phase) @ eigvecs.conj().T

    def _build_potential(self, pot_type: str) -> np.ndarray:
        if pot_type == "harmonic":
            k = 1.0
            return 0.5 * k * (self.x ** 2)
        elif pot_type == "barrier":
            v0 = self.config.barrier_height
            w = self.config.barrier_width
            return np.where(np.abs(self.x) < w, v0, 0.0)
        elif pot_type == "double_well":
            return 0.1 * (self.x ** 4 - 4 * self.x ** 2)
        else:  # free particle
            return np.zeros(self.N)

    def initialize_gaussian_wavepacket(self, x0: float = -1.0, sigma: float = 0.3, k0: float = 5.0) -> QuantumLatticeState:
        """Create normalized 1D Gaussian wavepacket ψ(x) = (1/(2πσ²)^(1/4)) * exp(-(x-x0)²/(4σ²)) * exp(i k0 x)."""
        norm_factor = (1.0 / (2.0 * np.pi * (sigma ** 2))) ** 0.25
        envelope = np.exp(-((self.x - x0) ** 2) / (4.0 * (sigma ** 2)))
        phase = np.exp(1j * k0 * self.x)
        psi = norm_factor * envelope * phase
        # Normalize on discrete lattice: sum |psi|^2 = 1
        psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2))
        return QuantumLatticeState.from_complex_array(psi=psi, step=0, time=0.0)

    def step(self, state: QuantumLatticeState) -> QuantumLatticeState:
        """Advance quantum state by one unitary timestep dt: ψ_(t+1) = U ψ_t."""
        psi = state.to_complex_array()
        psi_next = self.U @ psi
        return QuantumLatticeState.from_complex_array(
            psi=psi_next,
            step=state.step + 1,
            time=state.time + self.dt,
            branch_id=state.branch_id,
        )
