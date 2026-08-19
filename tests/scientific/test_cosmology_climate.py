"""Scientific Benchmark: Cosmological Angular Momentum Conservation & Climate Equilibrium (CHIMERA v5.0 - Phase 13)

Benchmark Goals:
1. Exact conservation of 3D angular momentum vector in closed gravitational N-body system (|ΔL/L| < 1e-4).
2. Stable global temperature attractor in 1D latitudinal energy balance climate model.
"""

import pytest
import numpy as np
from packages.cosmology.models import CelestialBody, CosmologicalWorldConfig
from packages.cosmology.nbody_cosmology import NBodyCosmologyEngine
from packages.cosmology.climate import PlanetaryClimateModel


def test_scientific_angular_momentum_conservation():
    star = CelestialBody(id="sun", name="Sun", mass=100.0, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))
    p1 = CelestialBody(id="p1", name="Planet1", mass=1.0, position=(5.0, 0.0, 0.0), velocity=(0.0, 4.47, 0.0))
    p2 = CelestialBody(id="p2", name="Planet2", mass=2.0, position=(0.0, 8.0, 0.0), velocity=(-3.53, 0.0, 0.0))

    bodies = [star, p1, p2]
    config = CosmologicalWorldConfig(g_grav=1.0, dt=0.005, softening=0.001)
    engine = NBodyCosmologyEngine(config)

    initial_metrics = engine.compute_energy_and_momentum(bodies)
    l_init = initial_metrics["angular_momentum_magnitude"]

    for _ in range(200):
        bodies = engine.step(bodies)

    final_metrics = engine.compute_energy_and_momentum(bodies)
    l_final = final_metrics["angular_momentum_magnitude"]
    delta_l_rel = abs(l_final - l_init) / l_init

    print(f"\n[Cosmology Benchmark] Initial L: {l_init:.6f} | Final L: {l_final:.6f} | Relative Drift: {delta_l_rel*100:.4f}%")

    assert delta_l_rel < 1e-4, f"Angular momentum drift {delta_l_rel} exceeded threshold"


def test_scientific_climate_attractor_stability():
    model = PlanetaryClimateModel(num_zones=18, dt_years=0.2)
    state = model.initialize_state(mean_temp_celsius=15.0)

    # Evolve 50 climate steps to reach equilibrium
    for _ in range(50):
        state = model.step(state)

    temps = np.array(state.temperatures)
    # Poles must be colder than equator
    equator_idx = len(temps) // 2
    equator_temp = temps[equator_idx]
    pole_temp = min(temps[0], temps[-1])

    print(f"\n[Climate Benchmark] Equator Temp: {equator_temp:.2f}K | North Pole Temp: {pole_temp:.2f}K")

    assert equator_temp > pole_temp + 20.0, "Equator must be significantly warmer than poles"
    assert 200.0 < pole_temp < 320.0, "Planetary temperatures must remain in realistic physical domain"
