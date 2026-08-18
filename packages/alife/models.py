"""Data Models for CHIMERA Artificial Life & Evolutionary Dynamics (Phase 7).

Defines schemas for:
  - Genome (speed_trait, perception_radius, metabolic_efficiency, reproduction_threshold, mutation_rate)
  - Organism (id, parent_id, generation, state, energy, age, genome)
  - Environment (nutrient food patches, spatial bounds, regeneration rate)
  - PhylogeneticNode (lineage DAG node)
  - EvolutionarySnapshot (time-series population state)
  - ALifeSimulationResult (complete evolutionary trajectory)
"""

from __future__ import annotations
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from packages.core.models import Vector2D, Boundary


class OrganismState(str, Enum):
    """Lifecycle state of an organism."""
    ALIVE = "alive"
    DEAD = "dead"
    REPRODUCING = "reproducing"


class Genome(BaseModel):
    """Evolvable genetic traits of an artificial organism."""
    model_config = ConfigDict(frozen=True)

    speed: float = Field(default=1.0, ge=0.1, le=5.0, description="Base movement speed")
    perception_radius: float = Field(default=20.0, ge=2.0, le=60.0, description="Vision sensing range for food")
    metabolic_efficiency: float = Field(default=1.0, ge=0.1, le=3.0, description="Energy conversion multiplier")
    reproduction_threshold: float = Field(default=35.0, ge=10.0, le=200.0, description="Energy required to reproduce")
    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Probability of trait mutation on birth")

    def mutate(self, rng: np.random.Generator) -> Genome:
        """Produce a mutated copy of this genome based on mutation_rate."""
        def perturb(val: float, low: float, high: float, scale: float = 0.1) -> float:
            if rng.uniform(0.0, 1.0) < self.mutation_rate:
                delta = rng.normal(0.0, scale * (high - low))
                return float(np.clip(val + delta, low, high))
            return val

        return Genome(
            speed=perturb(self.speed, 0.1, 5.0, 0.08),
            perception_radius=perturb(self.perception_radius, 2.0, 50.0, 0.08),
            metabolic_efficiency=perturb(self.metabolic_efficiency, 0.1, 3.0, 0.08),
            reproduction_threshold=perturb(self.reproduction_threshold, 10.0, 200.0, 0.08),
            mutation_rate=perturb(self.mutation_rate, 0.01, 0.3, 0.05),
        )

    def genetic_distance(self, other: Genome) -> float:
        """Normalized Euclidean distance between two genomes."""
        v1 = np.array([self.speed / 5.0, self.perception_radius / 50.0, self.metabolic_efficiency / 3.0, self.reproduction_threshold / 200.0])
        v2 = np.array([other.speed / 5.0, other.perception_radius / 50.0, other.metabolic_efficiency / 3.0, other.reproduction_threshold / 200.0])
        return float(np.linalg.norm(v1 - v2))


class Organism(BaseModel):
    """An individual cellular artificial organism."""
    model_config = ConfigDict(frozen=False)

    id: str = Field(default_factory=lambda: f"org_{uuid.uuid4().hex[:6]}")
    species_id: str = "sp_ancestor"
    parent_id: Optional[str] = None
    generation: int = 0
    position: Vector2D
    velocity: Vector2D = Field(default_factory=lambda: Vector2D(x=0.0, y=0.0))
    energy: float = 30.0
    age: int = 0
    state: OrganismState = OrganismState.ALIVE
    genome: Genome = Field(default_factory=Genome)
    offspring_count: int = 0


class FoodPatch(BaseModel):
    """Spatial nutrient resource item."""
    model_config = ConfigDict(frozen=True)

    id: int
    position: Vector2D
    energy_value: float = 15.0


class Environment(BaseModel):
    """Spatial environment with nutrient patches and boundaries."""
    model_config = ConfigDict(frozen=False)

    boundary: Boundary = Field(default_factory=lambda: Boundary(x_min=0.0, x_max=100.0, y_min=0.0, y_max=100.0))
    max_food: int = 40
    regeneration_rate: float = 0.5
    food_energy: float = 15.0
    food_patches: List[FoodPatch] = Field(default_factory=list)


class PhylogeneticNode(BaseModel):
    """A node in the phylogenetic evolutionary lineage tree."""
    model_config = ConfigDict(frozen=True)

    species_id: str
    parent_species_id: Optional[str]
    origin_generation: int
    extinction_generation: Optional[int] = None
    representative_genome: Genome
    total_offspring: int = 0


class EvolutionarySnapshot(BaseModel):
    """Population metrics at time step t."""
    model_config = ConfigDict(frozen=True)

    step: int
    time: float
    population_size: int
    food_count: int
    mean_energy: float
    mean_speed: float
    mean_perception: float
    mean_generation: float
    shannon_diversity: float
    active_species_count: int


class ALifeSimulationResult(BaseModel):
    """Complete time-series trajectory of an artificial life simulation."""
    model_config = ConfigDict(frozen=True)

    simulation_id: str = Field(default_factory=lambda: f"alife_{uuid.uuid4().hex[:8]}")
    total_steps: int
    total_births: int
    total_deaths: int
    snapshots: List[EvolutionarySnapshot]
    phylogenetic_tree_nodes: List[PhylogeneticNode]
    final_population_size: int
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
