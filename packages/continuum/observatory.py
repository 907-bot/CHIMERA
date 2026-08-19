"""Continuum Field Observatory & Metric Extraction (CHIMERA v2.0 - Phase 10)"""

from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
from packages.continuum.models import FluidState, FieldState, Grid2D


class ContinuumObservatory:
    """Extracts continuous field metrics, vorticity distributions, kinetic energy, and enstrophy."""

    def __init__(self, grid: Grid2D):
        self.grid = grid

    def compute_vorticity(self, state: FluidState) -> np.ndarray:
        """Compute 2D scalar vorticity field ω = ∂v/∂x - ∂u/∂y."""
        u, v, _ = state.to_numpy()
        dv_dx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * self.grid.dx)
        du_dy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * self.grid.dy)
        return dv_dx - du_dy

    def compute_metrics(self, state: FluidState) -> Dict[str, float]:
        """Compute global kinetic energy, enstrophy, and max vorticity."""
        u, v, p = state.to_numpy()
        vorticity = self.compute_vorticity(state)

        # Kinetic energy density = 0.5 * (u^2 + v^2)
        ke = 0.5 * np.mean(u ** 2 + v ** 2)
        # Enstrophy = 0.5 * integral(ω^2)
        enstrophy = 0.5 * np.mean(vorticity ** 2)
        max_speed = float(np.max(np.sqrt(u ** 2 + v ** 2)))
        max_vorticity = float(np.max(np.abs(vorticity)))

        # Incompressibility / Divergence check
        du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * self.grid.dx)
        dv_dy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * self.grid.dy)
        div = du_dx + dv_dy
        max_divergence = float(np.max(np.abs(div)))

        return {
            "step": float(state.step),
            "time": float(state.time),
            "kinetic_energy": float(ke),
            "enstrophy": float(enstrophy),
            "max_speed": max_speed,
            "max_vorticity": max_vorticity,
            "max_divergence": max_divergence,
        }
