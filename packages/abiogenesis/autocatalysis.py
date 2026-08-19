"""Autocatalytic Hypercycle & Catalytic Closure Engine (CHIMERA v4.0 - Phase 12)

Simulates Eigen-Schuster Hypercycles:
    dx_i/dt = k_i * x_i * x_{i-1} - Φ(x) * x_i
Where Φ(x) = sum(k_i * x_i * x_{i-1}) is the dilution flux maintaining constant total concentration.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Any
import numpy as np
import networkx as nx
from packages.abiogenesis.models import HypercycleState


class AutocatalyticHypercycleSolver:
    """Numerical solver for closed autocatalytic hypercycles and RAF sets."""

    def __init__(self, species_names: List[str], rate_constants: List[float], dt: float = 0.01):
        self.species_names = species_names
        self.n = len(species_names)
        self.k = np.array(rate_constants, dtype=np.float64)
        self.dt = dt

    def step(self, state: HypercycleState) -> HypercycleState:
        """Advance hypercycle dynamics using RK4 numerical integration."""
        x = np.array(state.species_concentrations, dtype=np.float64)

        def derivatives(c: np.ndarray) -> np.ndarray:
            # c_{i-1} cyclically rolls right
            c_prev = np.roll(c, 1)
            growth = self.k * c * c_prev
            # Total growth flux
            total_flux = np.sum(growth)
            total_c = np.sum(c)
            phi = total_flux / (total_c + 1e-12)
            dcdt = growth - phi * c
            return dcdt

        # Runge-Kutta 4th order
        k1 = derivatives(x)
        k2 = derivatives(x + 0.5 * self.dt * k1)
        k3 = derivatives(x + 0.5 * self.dt * k2)
        k4 = derivatives(x + self.dt * k3)

        x_next = x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Ensure non-negativity and normalization
        x_next = np.clip(x_next, 0.0, None)
        x_next = x_next / (np.sum(x_next) + 1e-12)

        return HypercycleState(
            step=state.step + 1,
            time=state.time + self.dt,
            species_concentrations=tuple(float(v) for v in x_next),
            species_names=state.species_names,
        )

    def detect_catalytic_closure(self, catalysis_matrix: np.ndarray) -> Dict[str, Any]:
        """Detects if catalysis graph contains a Reflexively Autocatalytic and Food-generated (RAF) set / cycle."""
        g = nx.DiGraph()
        n = catalysis_matrix.shape[0]
        for i in range(n):
            g.add_node(i)
            for j in range(n):
                if catalysis_matrix[i, j] > 0.0:
                    g.add_edge(i, j)

        cycles = list(nx.simple_cycles(g))
        has_closed_hypercycle = len(cycles) > 0

        return {
            "has_catalytic_closure": has_closed_hypercycle,
            "number_of_cycles": len(cycles),
            "cycle_paths": cycles,
            "strongly_connected_components": list(nx.strongly_connected_components(g)),
        }
