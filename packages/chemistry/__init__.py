"""CHIMERA Chemistry Package — Reaction Networks, Kinetics & Autocatalysis (Phase 6 — v0.6)

Provides deterministic mass-action chemical kinetics, hypergraph reaction cascades,
and automated detection of autocatalytic cycles and limit cycle chemical oscillators.

Modules:
  models      : Pydantic schemas for ChemicalSpecies, Reaction, ReactionNetwork, ChemistryState
  kinetics    : Deterministic Mass-Action ODE Solver (RK4 / Radau) & Canonical Benchmarks
  hypergraph  : Stoichiometric hypergraph & cycle topology analyzer
  detector    : Autonomous detector for autocatalysis and limit cycle oscillations
  agent       : ChemistAgent for reaction stoichiometry and pathway analysis
"""

from packages.chemistry.models import (
    ChemicalSpecies,
    Reaction,
    ReactionNetwork,
    ChemistryState,
    AutocatalyticCycleResult,
    KineticsSimulationResult,
)
from packages.chemistry.kinetics import MassActionKineticsSolver, BENCHMARK_NETWORKS
from packages.chemistry.hypergraph import ReactionHypergraph
from packages.chemistry.detector import AutocatalysisDetector
from packages.chemistry.agent import ChemistAgent

__all__ = [
    "ChemicalSpecies",
    "Reaction",
    "ReactionNetwork",
    "ChemistryState",
    "AutocatalyticCycleResult",
    "KineticsSimulationResult",
    "MassActionKineticsSolver",
    "BENCHMARK_NETWORKS",
    "ReactionHypergraph",
    "AutocatalysisDetector",
    "ChemistAgent",
]
