"""Multi-Scale Cosmic-to-Micro Planetary Orchestrator (CHIMERA v5.0 - Phase 13)"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from packages.cosmology.models import CelestialBody, ClimateGridState, MultiScaleState, CosmologicalWorldConfig
from packages.cosmology.nbody_cosmology import NBodyCosmologyEngine
from packages.cosmology.climate import PlanetaryClimateModel


class MultiScaleCosmologyOrchestrator:
    """Couples macroscopic celestial orbital mechanics with planetary climate dynamics."""

    def __init__(self, cosmo_config: CosmologicalWorldConfig):
        self.nbody_engine = NBodyCosmologyEngine(cosmo_config)
        self.climate_model = PlanetaryClimateModel()

    def step_multiscale(self, state: MultiScaleState) -> MultiScaleState:
        """Advance multi-scale system: updates orbits, adjusts insolation based on stellar distance, updates climate."""
        # 1. Step celestial orbits
        next_bodies = self.nbody_engine.step(state.bodies)

        # 2. Compute star-planet distance (assuming body 0 is star, body 1 is target planet)
        star = next_bodies[0]
        planet = next_bodies[1] if len(next_bodies) > 1 else next_bodies[0]

        r_vec = np.array(planet.position) - np.array(star.position)
        dist = float(np.linalg.norm(r_vec)) + 1e-5
        r_ref = 1.0  # reference 1 AU

        # Solar constant scales as 1/r^2
        adjusted_s0 = 1361.0 * ((r_ref / dist) ** 2)

        # Update climate with new solar constant
        climate_curr = state.climate.model_copy(update={"solar_constant": adjusted_s0})
        next_climate = self.climate_model.step(climate_curr)

        return MultiScaleState(
            cosmic_step=state.cosmic_step + 1,
            cosmic_time=state.cosmic_time + self.nbody_engine.dt,
            bodies=next_bodies,
            climate=next_climate,
            micro_substeps=state.micro_substeps,
        )
