"""Unit Tests for Phase 13: Planetary & Cosmological Scale Engine (CHIMERA v5.0)"""

import pytest
import numpy as np
from packages.cosmology.models import CelestialBody, CosmologicalWorldConfig, ClimateGridState
from packages.cosmology.nbody_cosmology import NBodyCosmologyEngine
from packages.cosmology.climate import PlanetaryClimateModel


def test_nbody_orbital_step():
    star = CelestialBody(id="star", name="Sun", mass=1000.0, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))
    planet = CelestialBody(id="planet", name="Earth", mass=1.0, position=(10.0, 0.0, 0.0), velocity=(0.0, 10.0, 0.0))
    config = CosmologicalWorldConfig(g_grav=1.0, dt=0.01, bodies=[star, planet])
    engine = NBodyCosmologyEngine(config)

    next_bodies = engine.step([star, planet])
    assert len(next_bodies) == 2
    assert next_bodies[1].position[1] > 0.0  # Planet moved along velocity direction


def test_planetary_climate_model():
    model = PlanetaryClimateModel(num_zones=10)
    state = model.initialize_state(mean_temp_celsius=15.0)
    assert len(state.temperatures) == 10

    state_next = model.step(state)
    assert state_next.step == 1
    assert len(state_next.temperatures) == 10
