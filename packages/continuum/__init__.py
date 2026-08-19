"""CHIMERA Continuum & Field Dynamics Engine (v2.0 - Phase 10)"""

from packages.continuum.models import Grid2D, ContinuumConfig, FluidState, FieldState
from packages.continuum.solvers import NavierStokesSolver2D, HeatSolver2D, WaveSolver2D
from packages.continuum.observatory import ContinuumObservatory
from packages.continuum.discovery import SpatialTemporalPDEDiscovery

__all__ = [
    "Grid2D",
    "ContinuumConfig",
    "FluidState",
    "FieldState",
    "NavierStokesSolver2D",
    "HeatSolver2D",
    "WaveSolver2D",
    "ContinuumObservatory",
    "SpatialTemporalPDEDiscovery",
]
