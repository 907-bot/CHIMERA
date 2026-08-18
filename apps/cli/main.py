"""CHIMERA Command Line Interface Application"""

import sys
import typer
from packages.core.models import WorldConfig
from packages.core.serialization import hash_world_state
from packages.physics.engine import DeterministicEngine
from packages.physics.energy import EnergyMetrics

app = typer.Typer(help="CHIMERA Artificial Multiverse Scientific Observatory CLI")


@app.command()
def run_sim(
    seed: int = typer.Option(42, help="Random seed for deterministic initialization"),
    steps: int = typer.Option(500, help="Number of simulation steps to execute"),
    particles: int = typer.Option(10, help="Number of particles in the universe"),
    integrator: str = typer.Option("verlet", help="Integrator type: verlet, rk4, euler"),
):
    """Run a deterministic physics simulation world and print state telemetry."""
    config = WorldConfig(
        seed=seed,
        num_particles=particles,
        integrator_type=integrator,
    )
    engine = DeterministicEngine(config=config)

    typer.echo(f"Initialized CHIMERA Engine (World={config.world_id}, Seed={seed}, Integrator={integrator})")
    
    initial_energy = EnergyMetrics.compute_all(engine.current_state.particles)
    typer.echo(f"Initial State Hash: {hash_world_state(engine.current_state)}")
    typer.echo(f"Initial Total Energy: {initial_energy['total_energy']:.6f}")

    history = engine.run(steps)
    final_state = history[-1]
    final_energy = EnergyMetrics.compute_all(final_state.particles)

    typer.echo(f"Completed {steps} steps.")
    typer.echo(f"Final State Hash:   {hash_world_state(final_state)}")
    typer.echo(f"Final Total Energy:   {final_energy['total_energy']:.6f}")
    
    drift = abs(final_energy['total_energy'] - initial_energy['total_energy']) / abs(initial_energy['total_energy'])
    typer.echo(f"Energy Drift (dE/E0): {drift:.8e}")


@app.command()
def verify_reproducibility(
    seed: int = typer.Option(42, help="Random seed for verification"),
    steps: int = typer.Option(500, help="Number of steps per verification run"),
):
    """Verify bit-for-bit trajectory reproducibility across independent runs."""
    config = WorldConfig(seed=seed)

    typer.echo(f"Executing Run A (Seed={seed}, Steps={steps})...")
    engine_a = DeterministicEngine(config=config)
    history_a = engine_a.run(steps)
    hash_a = hash_world_state(history_a[-1])

    typer.echo(f"Executing Run B (Seed={seed}, Steps={steps})...")
    engine_b = DeterministicEngine(config=config)
    history_b = engine_b.run(steps)
    hash_b = hash_world_state(history_b[-1])

    typer.echo(f"Run A Final Hash: {hash_a}")
    typer.echo(f"Run B Final Hash: {hash_b}")

    if hash_a == hash_b:
        typer.secho("SUCCESS: Bitwise reproducibility verified!", fg=typer.colors.GREEN, bold=True)
    else:
        typer.secho("FAILURE: State hashes do not match!", fg=typer.colors.RED, bold=True)
        sys.exit(1)


if __name__ == "__main__":
    app()
