"""Vectorized PDE Numerical Solvers for Continuum Mechanics (CHIMERA v2.0 - Phase 10)

Equations Implemented:
1. Incompressible Navier-Stokes:
   ∂u/∂t + (u·∇)u = - (1/ρ) ∇p + ν ∇²u
   ∇ · u = 0 (Incompressibility condition via Chorin Projection)

2. 2D Heat / Diffusion Equation:
   ∂T/∂t = α ∇²T

3. 2D Wave / Electrodynamics Equation:
   ∂²E/∂t² = c² ∇²E
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from packages.continuum.models import ContinuumConfig, Grid2D, FluidState, FieldState


class NavierStokesSolver2D:
    """Vectorized 2D Incompressible Navier-Stokes Solver using Chorin Projection Method."""

    def __init__(self, config: ContinuumConfig):
        self.config = config
        self.grid = config.grid
        self.nx = self.grid.nx
        self.ny = self.grid.ny
        self.dx = self.grid.dx
        self.dy = self.grid.dy
        self.dt = self.config.dt
        self.nu = self.config.viscosity
        self.rho = self.config.density

    def laplacian(self, f: np.ndarray) -> np.ndarray:
        """Compute 2D periodic discrete Laplacian ∇²f."""
        lap = (
            (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / (self.dx ** 2)
            + (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / (self.dy ** 2)
        )
        return lap

    def gradient(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute periodic central difference gradient (df/dx, df/dy)."""
        df_dx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * self.dx)
        df_dy = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * self.dy)
        return df_dx, df_dy

    def divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute divergence ∇ · (u, v)."""
        du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * self.dx)
        dv_dy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * self.dy)
        return du_dx + dv_dy

    def solve_pressure_poisson(self, div_star: np.ndarray, max_iter: int = 50, tol: float = 1e-5) -> np.ndarray:
        """Solve Poisson equation ∇²p = (ρ/dt) * (∇ · u*) using Jacobi iterations."""
        rhs = (self.rho / self.dt) * div_star
        p = np.zeros_like(div_star)
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        denom = 2.0 * (dx2 + dy2)

        for _ in range(max_iter):
            p_next = (
                (np.roll(p, -1, axis=1) + np.roll(p, 1, axis=1)) * dy2
                + (np.roll(p, -1, axis=0) + np.roll(p, 1, axis=0)) * dx2
                - rhs * dx2 * dy2
            ) / denom
            diff = np.max(np.abs(p_next - p))
            p = p_next
            if diff < tol:
                break
        return p

    def step(self, state: FluidState) -> FluidState:
        """Advance fluid state by one deterministic timestep dt."""
        u, v, p = state.to_numpy()

        # 1. Advection terms (u·∇)u and (u·∇)v
        du_dx, du_dy = self.gradient(u)
        dv_dx, dv_dy = self.gradient(v)
        adv_u = u * du_dx + v * du_dy
        adv_v = u * dv_dx + v * dv_dy

        # 2. Diffusion terms ν ∇²u and ν ∇²v
        diff_u = self.nu * self.laplacian(u)
        diff_v = self.nu * self.laplacian(v)

        # Intermediate velocity fields u*, v*
        u_star = u + self.dt * (-adv_u + diff_u)
        v_star = v + self.dt * (-adv_v + diff_v)

        # 3. Pressure Poisson equation for incompressibility
        div_star = self.divergence(u_star, v_star)
        p_new = self.solve_pressure_poisson(div_star)

        # 4. Project intermediate velocity to divergence-free field: u^(n+1) = u* - (dt/ρ) ∇p
        dp_dx, dp_dy = self.gradient(p_new)
        u_next = u_star - (self.dt / self.rho) * dp_dx
        v_next = v_star - (self.dt / self.rho) * dp_dy

        return FluidState.from_numpy(
            step=state.step + 1,
            time=state.time + self.dt,
            u=u_next,
            v=v_next,
            p=p_new,
        )


class HeatSolver2D:
    """Vectorized 2D Heat/Diffusion Equation Solver (∂T/∂t = α ∇²T)."""

    def __init__(self, config: ContinuumConfig):
        self.config = config
        self.alpha = config.thermal_diffusivity
        self.dx = config.grid.dx
        self.dy = config.grid.dy
        self.dt = config.dt

    def step(self, state: FieldState) -> FieldState:
        T = state.to_numpy()
        lap_T = (
            (np.roll(T, -1, axis=1) - 2 * T + np.roll(T, 1, axis=1)) / (self.dx ** 2)
            + (np.roll(T, -1, axis=0) - 2 * T + np.roll(T, 1, axis=0)) / (self.dy ** 2)
        )
        T_next = T + self.dt * self.alpha * lap_T
        return FieldState.from_numpy(
            step=state.step + 1,
            time=state.time + self.dt,
            data=T_next,
            field_name=state.field_name,
        )


class WaveSolver2D:
    """Vectorized 2D Maxwell/Wave Equation Solver (∂²E/∂t² = c² ∇²E)."""

    def __init__(self, config: ContinuumConfig):
        self.config = config
        self.c = config.c_light
        self.dx = config.grid.dx
        self.dy = config.grid.dy
        self.dt = config.dt

    def step_wave(self, u_curr: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        lap = (
            (np.roll(u_curr, -1, axis=1) - 2 * u_curr + np.roll(u_curr, 1, axis=1)) / (self.dx ** 2)
            + (np.roll(u_curr, -1, axis=0) - 2 * u_curr + np.roll(u_curr, 1, axis=0)) / (self.dy ** 2)
        )
        u_next = 2 * u_curr - u_prev + (self.c * self.dt) ** 2 * lap
        return u_next
