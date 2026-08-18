"""Evolutionary Engine & Phylogenetic Lineage Tracker for CHIMERA Phase 7.

Simulates multi-generation natural selection, carrying-capacity competition,
speciation dynamics, and records the phylogenetic lineage DAG.
"""

from __future__ import annotations
import math
import numpy as np
import networkx as nx
from typing import List, Dict, Set, Optional, Tuple
from packages.core.models import Vector2D, Boundary
from packages.alife.models import (
    Organism,
    OrganismState,
    Environment,
    FoodPatch,
    Genome,
    PhylogeneticNode,
    EvolutionarySnapshot,
    ALifeSimulationResult,
)
from packages.alife.metabolism import MetabolicEngine


class EvolutionaryEngine:
    """Simulates artificial life populations, speciation, and phylogenetic lineage trees."""

    def __init__(
        self,
        seed: int = 42,
        speciation_threshold: float = 0.35,
        basal_burn_rate: float = 0.05,
    ):
        self.rng = np.random.default_rng(seed)
        self.speciation_threshold = speciation_threshold
        self.metabolism = MetabolicEngine(basal_burn_rate=basal_burn_rate)
        self.phylogenetic_tree: Dict[str, PhylogeneticNode] = {}
        self.next_species_idx = 1
        self.next_food_id = 1

    def _initialize_environment(self, env: Environment) -> Environment:
        """Populate initial food patches."""
        b = env.boundary
        patches = []
        for _ in range(env.max_food):
            fx = float(self.rng.uniform(b.x_min + 5.0, b.x_max - 5.0))
            fy = float(self.rng.uniform(b.y_min + 5.0, b.y_max - 5.0))
            patches.append(FoodPatch(id=self.next_food_id, position=Vector2D(x=fx, y=fy), energy_value=env.food_energy))
            self.next_food_id += 1
        env.food_patches = patches
        return env

    def _regenerate_food(self, env: Environment) -> None:
        """Regenerate food up to carrying capacity."""
        b = env.boundary
        needed = env.max_food - len(env.food_patches)
        if needed > 0 and self.rng.uniform(0.0, 1.0) < env.regeneration_rate:
            spawn_count = min(needed, int(self.rng.integers(1, 4)))
            for _ in range(spawn_count):
                fx = float(self.rng.uniform(b.x_min + 5.0, b.x_max - 5.0))
                fy = float(self.rng.uniform(b.y_min + 5.0, b.y_max - 5.0))
                env.food_patches.append(
                    FoodPatch(id=self.next_food_id, position=Vector2D(x=fx, y=fy), energy_value=env.food_energy)
                )
                self.next_food_id += 1

    def _handle_speciation(self, offspring: Organism, parent_species_id: str, generation: int) -> str:
        """Assign offspring to existing species or declare a new species branch."""
        parent_node = self.phylogenetic_tree.get(parent_species_id)
        if parent_node is None:
            # Root ancestor
            sp_id = "sp_ancestor"
            self.phylogenetic_tree[sp_id] = PhylogeneticNode(
                species_id=sp_id,
                parent_species_id=None,
                origin_generation=0,
                representative_genome=offspring.genome,
                total_offspring=1,
            )
            return sp_id

        # Measure genetic distance to parent species prototype
        dist = offspring.genome.genetic_distance(parent_node.representative_genome)
        if dist > self.speciation_threshold:
            # Speciation event: fork new branch in lineage DAG
            new_sp_id = f"sp_{self.next_species_idx:03d}"
            self.next_species_idx += 1

            self.phylogenetic_tree[new_sp_id] = PhylogeneticNode(
                species_id=new_sp_id,
                parent_species_id=parent_species_id,
                origin_generation=generation,
                representative_genome=offspring.genome,
                total_offspring=1,
            )
            return new_sp_id
        else:
            # Retain parent species
            # Update parent offspring counter
            updated_parent = parent_node.model_copy(update={"total_offspring": parent_node.total_offspring + 1})
            self.phylogenetic_tree[parent_species_id] = updated_parent
            return parent_species_id

    def run_simulation(
        self,
        initial_population_size: int = 15,
        total_steps: int = 150,
        dt: float = 1.0,
        env: Optional[Environment] = None,
    ) -> ALifeSimulationResult:
        """Run evolutionary dynamics simulation for total_steps."""
        environment = env or Environment()
        self._initialize_environment(environment)

        b = environment.boundary
        population: List[Organism] = []

        # Create root ancestor node
        root_genome = Genome()
        self.phylogenetic_tree["sp_ancestor"] = PhylogeneticNode(
            species_id="sp_ancestor",
            parent_species_id=None,
            origin_generation=0,
            representative_genome=root_genome,
            total_offspring=initial_population_size,
        )

        for _ in range(initial_population_size):
            ox = float(self.rng.uniform(b.x_min + 10.0, b.x_max - 10.0))
            oy = float(self.rng.uniform(b.y_min + 10.0, b.y_max - 10.0))
            population.append(
                Organism(
                    species_id="sp_ancestor",
                    position=Vector2D(x=ox, y=oy),
                    energy=35.0,
                    genome=root_genome,
                )
            )

        snapshots: List[EvolutionarySnapshot] = []
        total_births = 0
        total_deaths = 0

        for step in range(total_steps):
            alive_organisms: List[Organism] = []
            new_offspring: List[Organism] = []
            consumed_food_ids: Set[int] = set()

            for org in population:
                updated_org, offspring, food_id = self.metabolism.update_organism(
                    org, environment, dt, self.rng
                )
                if food_id is not None:
                    consumed_food_ids.add(food_id)

                if updated_org.state == OrganismState.ALIVE:
                    alive_organisms.append(updated_org)
                else:
                    total_deaths += 1

                if offspring is not None:
                    total_births += 1
                    child_sp = self._handle_speciation(
                        offspring, updated_org.species_id, offspring.generation
                    )
                    offspring.species_id = child_sp
                    new_offspring.append(offspring)

            # Remove eaten food and regenerate
            environment.food_patches = [f for f in environment.food_patches if f.id not in consumed_food_ids]
            self._regenerate_food(environment)

            # Next generation population
            population = alive_organisms + new_offspring

            # Population statistics
            pop_size = len(population)
            if pop_size > 0:
                mean_energy = float(np.mean([o.energy for o in population]))
                mean_speed = float(np.mean([o.genome.speed for o in population]))
                mean_perception = float(np.mean([o.genome.perception_radius for o in population]))
                mean_gen = float(np.mean([o.generation for o in population]))

                # Shannon diversity of active species
                species_counts: Dict[str, int] = {}
                for o in population:
                    species_counts[o.species_id] = species_counts.get(o.species_id, 0) + 1
                
                shannon_h = 0.0
                for count in species_counts.values():
                    p_i = count / pop_size
                    shannon_h -= p_i * math.log(p_i + 1e-12)

                active_species = len(species_counts)
            else:
                mean_energy = 0.0
                mean_speed = 0.0
                mean_perception = 0.0
                mean_gen = 0.0
                shannon_h = 0.0
                active_species = 0

            snapshots.append(
                EvolutionarySnapshot(
                    step=step,
                    time=round(step * dt, 4),
                    population_size=pop_size,
                    food_count=len(environment.food_patches),
                    mean_energy=round(mean_energy, 4),
                    mean_speed=round(mean_speed, 4),
                    mean_perception=round(mean_perception, 4),
                    mean_generation=round(mean_gen, 4),
                    shannon_diversity=round(shannon_h, 4),
                    active_species_count=active_species,
                )
            )

        return ALifeSimulationResult(
            total_steps=total_steps,
            total_births=total_births,
            total_deaths=total_deaths,
            snapshots=snapshots,
            phylogenetic_tree_nodes=list(self.phylogenetic_tree.values()),
            final_population_size=len(population),
        )
