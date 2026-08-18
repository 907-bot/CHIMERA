"""Deterministic Physics Simulation Engine Orchestrator"""

from typing import List, Optional
import numpy as np
from packages.core.models import (
    WorldState,
    WorldConfig,
    Particle,
    Vector2D,
    Boundary,
)
from packages.core.serialization import hash_world_config, hash_world_state
from packages.physics.forces import ForceField, GravityForce, HarmonicForce, DragForce
from packages.physics.collisions import BoundaryCollision, ParticleCollision
from packages.physics.integrators import (
    BaseIntegrator,
    VerletIntegrator,
    RK4Integrator,
    EulerIntegrator,
)


class DeterministicEngine:
    """Deterministic 2D Physics Engine maintaining Bitwise Reproducibility."""

    def __init__(self, config: Optional[WorldConfig] = None):
        self.config = config or WorldConfig()
        self.config_hash = hash_world_config(self.config)

        # Force field setup based on config
        self.force_field = ForceField()
        if self.config.gravity_constant != 0:
            self.force_field.add_force(
                GravityForce(G=self.config.gravity_constant, softening=self.config.softening)
            )

        # Collision handlers
        self.boundary_collision = BoundaryCollision(restitution=self.config.restitution)
        self.particle_collision = ParticleCollision(restitution=self.config.restitution)

        # Integrator selection
        integrator_type = self.config.integrator_type.lower()
        if integrator_type == "rk4":
            self.integrator: BaseIntegrator = RK4Integrator()
        elif integrator_type == "euler":
            self.integrator = EulerIntegrator()
        else:
            self.integrator = VerletIntegrator()

        # Initialize or set current world state
        self.current_state = self._initialize_state()

    def _initialize_state(self) -> WorldState:
        """Create deterministic initial state using explicit seed sequence."""
        rng = np.random.default_rng(self.config.seed)

        particles = []
        b = self.config.boundary

        # Margin to prevent immediate boundary overlap
        margin = 5.0
        x_min, x_max = b.x_min + margin, b.x_max - margin
        y_min, y_max = b.y_min + margin, b.y_max - margin

        for i in range(self.config.num_particles):
            px = float(rng.uniform(x_min, x_max))
            py = float(rng.uniform(y_min, y_max))
            vx = float(rng.uniform(-2.0, 2.0))
            vy = float(rng.uniform(-2.0, 2.0))

            p = Particle(
                id=i + 1,
                mass=1.0,
                radius=1.0,
                position=Vector2D(x=px, y=py),
                velocity=Vector2D(x=vx, y=vy),
            )
            particles.append(p)

        return WorldState(
            world_id=self.config.world_id,
            step=0,
            time=0.0,
            dt=self.config.dt,
            particles=particles,
            boundary=self.config.boundary,
            seed=self.config.seed,
            config_hash=self.config_hash,
        )

    def step(self) -> WorldState:
        """Advance world state by 1 deterministic step."""
        state = self.current_state

        # 1. Integrate positions and velocities
        updated_particles = self.integrator.step(
            particles=state.particles,
            dt=state.dt,
            force_field=self.force_field,
        )

        # 2. Resolve particle-particle collisions
        updated_particles = self.particle_collision.resolve(updated_particles)

        # 3. Resolve boundary collisions
        updated_particles = self.boundary_collision.resolve(updated_particles, state.boundary)

        # 4. Construct next state
        next_step = state.step + 1
        next_time = round(state.time + state.dt, 10)

        next_state = WorldState(
            world_id=state.world_id,
            step=next_step,
            time=next_time,
            dt=state.dt,
            particles=updated_particles,
            boundary=state.boundary,
            seed=state.seed,
            config_hash=state.config_hash,
        )

        self.current_state = next_state
        return next_state

    def run(self, num_steps: int) -> List[WorldState]:
        """Run simulation for num_steps and return list of state snapshots."""
        history = [self.current_state]
        for _ in range(num_steps):
            s = self.step()
            history.append(s)
        return history

    def restore_state(self, state: WorldState):
        """Restore engine state to a specific checkpoint state."""
        self.current_state = state
