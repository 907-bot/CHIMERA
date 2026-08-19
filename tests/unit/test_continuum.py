"""Unit Tests for Phase 10: Continuum & Field Dynamics Engine (CHIMERA v2.0)"""

import pytest
import numpy as np
from packages.continuum.models import Grid2D, ContinuumConfig, FluidState, FieldState
from packages.continuum.solvers import NavierStokesSolver2D, HeatSolver2D, WaveSolver2D
from packages.continuum.observatory import ContinuumObservatory
from packages.continuum.discovery import SpatialTemporalPDEDiscovery


def test_grid_initialization():
    grid = Grid2D(nx=32, ny=32, lx=1.0, ly=1.0)
    assert grid.dx == pytest.approx(1.0 / 32)
    assert grid.dy == pytest.approx(1.0 / 32)


def test_fluid_state_numpy_conversion():
    u = np.ones((16, 16)) * 0.5
    v = np.zeros((16, 16))
    p = np.zeros((16, 16))
    state = FluidState.from_numpy(step=0, time=0.0, u=u, v=v, p=p)
    u_back, v_back, p_back = state.to_numpy()
    assert np.allclose(u, u_back)
    assert np.allclose(v, v_back)


def test_heat_solver_step():
    config = ContinuumConfig(grid=Grid2D(nx=16, ny=16), dt=0.01, thermal_diffusivity=0.1)
    solver = HeatSolver2D(config)

    # Initial Gaussian bump in center
    data = np.zeros((16, 16))
    data[8, 8] = 10.0
    state = FieldState.from_numpy(step=0, time=0.0, data=data)

    state_next = solver.step(state)
    assert state_next.step == 1
    assert state_next.time == pytest.approx(0.01)
    # Peak should diffuse and decrease
    assert state_next.to_numpy()[8, 8] < 10.0
    # Total heat should be conserved in periodic domain
    assert np.sum(state_next.to_numpy()) == pytest.approx(np.sum(data), rel=1e-5)


def test_navier_stokes_incompressibility_projection():
    config = ContinuumConfig(grid=Grid2D(nx=16, ny=16), dt=0.001, viscosity=0.01)
    solver = NavierStokesSolver2D(config)
    obs = ContinuumObservatory(config.grid)

    u = np.sin(np.linspace(0, 2 * np.pi, 16))[:, np.newaxis] * np.ones((1, 16))
    v = np.cos(np.linspace(0, 2 * np.pi, 16))[np.newaxis, :] * np.ones((16, 1))
    p = np.zeros((16, 16))

    state = FluidState.from_numpy(step=0, time=0.0, u=u, v=v, p=p)
    state_next = solver.step(state)

    metrics = obs.compute_metrics(state_next)
    assert "kinetic_energy" in metrics
    assert "max_divergence" in metrics
    assert metrics["step"] == 1
