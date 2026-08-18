"""CHIMERA Artificial Life & Evolutionary Dynamics Package (Phase 7 — v0.7 / v0.8)

Provides cellular organisms with internal metabolism, genetic inheritance,
mutations, natural selection, speciation, and phylogenetic lineage tracking.

Modules:
  models      : Genome, Organism, Environment, PhylogeneticNode, EvolutionarySnapshot
  metabolism  : MetabolicEngine (energy budgets, foraging, basal costs, reproduction)
  evolution   : EvolutionaryEngine (population dynamics, speciation, phylogenetic tree)
  agent       : BiologistAgent (analyzing convergent evolution and lineage metrics)
"""

from packages.alife.models import (
    Genome,
    Organism,
    Environment,
    PhylogeneticNode,
    EvolutionarySnapshot,
    ALifeSimulationResult,
)
from packages.alife.metabolism import MetabolicEngine
from packages.alife.evolution import EvolutionaryEngine
from packages.alife.agent import BiologistAgent

__all__ = [
    "Genome",
    "Organism",
    "Environment",
    "PhylogeneticNode",
    "EvolutionarySnapshot",
    "ALifeSimulationResult",
    "MetabolicEngine",
    "EvolutionaryEngine",
    "BiologistAgent",
]
