"""Metabolic Engine for Cellular Organisms in CHIMERA Phase 7.

Handles energy budgets, sensory foraging, movement mechanics, feeding,
metabolic burn, senescence, and mitotic division with mutation.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from packages.core.models import Vector2D, Boundary
from packages.alife.models import Organism, OrganismState, Environment, FoodPatch, Genome


class MetabolicEngine:
    """Simulates internal bioenergetics and physical behavior of artificial organisms."""

    def __init__(self, basal_burn_rate: float = 0.05, max_age: int = 1000):
        self.basal_burn_rate = basal_burn_rate
        self.max_age = max_age

    def update_organism(
        self,
        organism: Organism,
        env: Environment,
        dt: float,
        rng: np.random.Generator,
    ) -> Tuple[Organism, Optional[Organism], Optional[int]]:
        """Advance one organism by time dt.

        Returns:
            Tuple of (updated_organism, optional_offspring, consumed_food_id).
        """
        if organism.state == OrganismState.DEAD:
            return organism, None, None

        # 1. Sensory perception: Find closest food patch within perception radius
        g = organism.genome
        closest_food: Optional[FoodPatch] = None
        min_dist = float("inf")

        for food in env.food_patches:
            dist = math.hypot(food.position.x - organism.position.x, food.position.y - organism.position.y)
            if dist <= g.perception_radius and dist < min_dist:
                min_dist = dist
                closest_food = food

        # 2. Movement & steering
        if closest_food is not None:
            # Steer directly toward food
            dx = closest_food.position.x - organism.position.x
            dy = closest_food.position.y - organism.position.y
            angle = math.atan2(dy, dx)
        else:
            # Random wander
            angle = rng.uniform(0.0, 2.0 * math.pi)

        vx = g.speed * math.cos(angle)
        vy = g.speed * math.sin(angle)
        organism.velocity = Vector2D(x=vx, y=vy)

        # Update position with boundary bounce/clamp
        b = env.boundary
        new_x = float(np.clip(organism.position.x + vx * dt, b.x_min, b.x_max))
        new_y = float(np.clip(organism.position.y + vy * dt, b.y_min, b.y_max))
        organism.position = Vector2D(x=new_x, y=new_y)

        # 3. Metabolic cost calculation
        # Higher speed and larger perception cost more baseline energy
        metabolic_burn = (self.basal_burn_rate * (g.speed ** 1.5) + 0.01 * (g.perception_radius / 10.0)) * dt
        organism.energy -= metabolic_burn
        organism.age += 1

        # 4. Foraging & Feeding
        consumed_food_id: Optional[int] = None
        if closest_food is not None and min_dist <= 2.5:
            # Consume food
            energy_gain = closest_food.energy_value * g.metabolic_efficiency
            organism.energy += energy_gain
            consumed_food_id = closest_food.id

        # 5. Check Mortality (Starvation or Old Age)
        if organism.energy <= 0.0 or organism.age >= self.max_age:
            organism.state = OrganismState.DEAD
            return organism, None, consumed_food_id

        # 6. Reproduction (Mitosis when energy exceeds threshold)
        offspring: Optional[Organism] = None
        if organism.energy >= g.reproduction_threshold:
            # Split energy
            organism.energy *= 0.5
            organism.offspring_count += 1

            # Mutate child genome
            child_genome = g.mutate(rng)
            
            # Spawn slightly offset from parent
            spawn_dx = float(rng.uniform(-1.0, 1.0))
            spawn_dy = float(rng.uniform(-1.0, 1.0))

            offspring = Organism(
                species_id=organism.species_id,
                parent_id=organism.id,
                generation=organism.generation + 1,
                position=Vector2D(
                    x=float(np.clip(organism.position.x + spawn_dx, b.x_min, b.x_max)),
                    y=float(np.clip(organism.position.y + spawn_dy, b.y_min, b.y_max)),
                ),
                velocity=Vector2D(x=0.0, y=0.0),
                energy=organism.energy,
                age=0,
                state=OrganismState.ALIVE,
                genome=child_genome,
            )

        return organism, offspring, consumed_food_id
