"""Scientific Benchmark: Energy Conservation and Numerical Integration Drift"""

import pytest
from packages.core.models import WorldConfig
from packages.physics.engine import DeterministicEngine
from packages.physics.energy import EnergyMetrics


def test_verlet_energy_conservation():
    """Verify Velocity-Verlet energy drift ratio |dE|/E0 remains < 1e-4 over 1000 steps."""
    config = WorldConfig(
        seed=101,
        num_particles=8,
        dt=0.002,
        integrator_type="verlet",
        gravity_constant=1.0,
        softening=0.5,
        restitution=1.0,  # Perfectly elastic
    )
    engine = DeterministicEngine(config=config)

    initial_energy = EnergyMetrics.compute_all(engine.current_state.particles, G=config.gravity_constant, softening=config.softening)
    e0 = abs(initial_energy["total_energy"])

    # Run for 1,000 steps
    history = engine.run(1000)
    final_energy = EnergyMetrics.compute_all(history[-1].particles, G=config.gravity_constant, softening=config.softening)
    ef = final_energy["total_energy"]

    energy_drift = abs(ef - initial_energy["total_energy"]) / e0
    assert energy_drift < 1e-4, f"Verlet energy drift ratio {energy_drift:.8e} exceeded threshold 1e-4"


def test_verlet_vs_euler_energy_drift():
    """Demonstrate that Symplectic Verlet maintains vastly superior energy conservation compared to Euler."""
    config_verlet = WorldConfig(seed=202, num_particles=6, dt=0.005, integrator_type="verlet", softening=0.5)
    config_euler = WorldConfig(seed=202, num_particles=6, dt=0.005, integrator_type="euler", softening=0.5)

    engine_verlet = DeterministicEngine(config=config_verlet)
    engine_euler = DeterministicEngine(config=config_euler)

    e0_v = abs(EnergyMetrics.compute_all(engine_verlet.current_state.particles, softening=0.5)["total_energy"])
    e0_e = abs(EnergyMetrics.compute_all(engine_euler.current_state.particles, softening=0.5)["total_energy"])

    h_v = engine_verlet.run(500)
    h_e = engine_euler.run(500)

    ef_v = EnergyMetrics.compute_all(h_v[-1].particles, softening=0.5)["total_energy"]
    ef_e = EnergyMetrics.compute_all(h_e[-1].particles, softening=0.5)["total_energy"]

    drift_v = abs(ef_v - e0_v) / e0_v
    drift_e = abs(ef_e - e0_e) / e0_e

    # Verlet drift must be at least 1 order of magnitude smaller than Euler drift
    assert drift_v < drift_e
