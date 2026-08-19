"""Spatial-Temporal PDE Discovery Engine (CHIMERA v2.0 - Phase 10)

Recovers governing PDE parameters (e.g. thermal diffusivity α, fluid advection, diffusion)
from spatial-temporal continuous field grids using sparse linear regression (PDE-FIND / PySINDy approach).
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np


class SpatialTemporalPDEDiscovery:
    """Discovers PDE governing coefficients from field history snapshots."""

    def __init__(self, dx: float, dy: float, dt: float):
        self.dx = dx
        self.dy = dy
        self.dt = dt

    def discover_heat_equation(self, field_snapshots: List[np.ndarray]) -> Dict[str, Any]:
        """Discovers parameter α in ∂T/∂t = α ∇²T from snapshot history."""
        if len(field_snapshots) < 2:
            raise ValueError("Need at least 2 consecutive snapshots for time derivative")

        u_t_list = []
        lap_list = []

        for i in range(len(field_snapshots) - 1):
            T_curr = field_snapshots[i]
            T_next = field_snapshots[i + 1]

            # Time derivative dT/dt
            dt_field = (T_next - T_curr) / self.dt

            # Spatial Laplacian ∇²T
            lap = (
                (np.roll(T_curr, -1, axis=1) - 2 * T_curr + np.roll(T_curr, 1, axis=1)) / (self.dx ** 2)
                + (np.roll(T_curr, -1, axis=0) - 2 * T_curr + np.roll(T_curr, 1, axis=0)) / (self.dy ** 2)
            )

            u_t_list.append(dt_field.flatten())
            lap_list.append(lap.flatten())

        y = np.concatenate(u_t_list)
        X = np.concatenate(lap_list)[:, np.newaxis]

        # Least squares regression: y = α * X
        alpha_discovered, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        alpha_est = float(alpha_discovered[0])

        # R² score calculation
        y_pred = X * alpha_est
        ss_res = np.sum((y - y_pred.flatten()) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
        r2 = 1.0 - (ss_res / ss_tot)

        return {
            "discovered_pde": f"∂T/∂t = {alpha_est:.6f} * ∇²T",
            "diffusivity_alpha": alpha_est,
            "r2_score": float(r2),
            "num_samples": len(y),
        }
