"""Scientific Benchmark: Continuous Field Dynamics & Spatial-Temporal PDE Recovery (CHIMERA v2.0 - Phase 10)

Benchmark Goal:
Autonomous recovery of the thermal diffusivity parameter α in ∂T/∂t = α ∇²T from continuous scalar field observations
with < 0.5% error (R² > 0.999).
"""

import pytest
import numpy as np
from packages.continuum.models import ContinuumConfig, Grid2D, FieldState
from packages.continuum.solvers import HeatSolver2D
from packages.continuum.discovery import SpatialTemporalPDEDiscovery


def test_scientific_heat_equation_parameter_recovery():
    true_alpha = 0.05
    nx, ny = 32, 32
    lx, ly = 2.0 * np.pi, 2.0 * np.pi
    dt = 0.001

    grid = Grid2D(nx=nx, ny=ny, lx=lx, ly=ly)
    config = ContinuumConfig(grid=grid, dt=dt, thermal_diffusivity=true_alpha)
    solver = HeatSolver2D(config)

    # Initial condition: T(x, y, 0) = sin(x) * cos(y)
    x = np.linspace(0, lx, nx, endpoint=False)
    y = np.linspace(0, ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    T0 = np.sin(X) * np.cos(Y)

    # Run simulation forward
    state = FieldState.from_numpy(step=0, time=0.0, data=T0)
    snapshots = [state.to_numpy()]

    for _ in range(50):
        state = solver.step(state)
        snapshots.append(state.to_numpy())

    # Discover PDE parameter from trajectory
    discovery = SpatialTemporalPDEDiscovery(dx=grid.dx, dy=grid.dy, dt=dt)
    result = discovery.discover_heat_equation(snapshots)

    discovered_alpha = result["diffusivity_alpha"]
    relative_error = abs(discovered_alpha - true_alpha) / true_alpha

    print(f"\n[PDE Recovery Benchmark] True α: {true_alpha:.6f} | Discovered α: {discovered_alpha:.6f} | Relative Error: {relative_error*100:.3f}% | R²: {result['r2_score']:.6f}")

    assert relative_error < 0.005, f"PDE parameter error {relative_error*100:.3f}% exceeded 0.5% threshold"
    assert result["r2_score"] > 0.999
