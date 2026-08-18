"""High-Precision Numerical Integrators for Deterministic Physics Engine"""

from abc import ABC, abstractmethod
from typing import List
from packages.core.models import Particle, Vector2D
from packages.physics.forces import ForceField


class BaseIntegrator(ABC):
    """Abstract Base Integrator interface."""

    @abstractmethod
    def step(self, particles: List[Particle], dt: float, force_field: ForceField) -> List[Particle]:
        """Advance system state by time step dt."""
        pass


class EulerIntegrator(BaseIntegrator):
    """Explicit 1st-Order Forward Euler Integrator."""

    def step(self, particles: List[Particle], dt: float, force_field: ForceField) -> List[Particle]:
        forces = force_field.compute_net_forces(particles)
        updated = []
        for p, f in zip(particles, forces):
            acc = f / p.mass
            new_pos = p.position + (p.velocity * dt)
            new_vel = p.velocity + (acc * dt)
            updated.append(p.with_position(new_pos).with_velocity(new_vel).with_force(f))
        return updated


class VerletIntegrator(BaseIntegrator):
    """Symplectic Velocity-Verlet Integrator for Energy-Conserving Hamiltonian Systems."""

    def step(self, particles: List[Particle], dt: float, force_field: ForceField) -> List[Particle]:
        # 1. Compute forces at current position
        f_current = force_field.compute_net_forces(particles)

        # 2. Half-step velocity update & full-step position update
        half_step_particles = []
        for p, f in zip(particles, f_current):
            acc = f / p.mass
            half_vel = p.velocity + (0.5 * dt * acc)
            new_pos = p.position + (dt * half_vel)
            half_step_particles.append(p.with_position(new_pos).with_velocity(half_vel))

        # 3. Compute forces at new position
        f_new = force_field.compute_net_forces(half_step_particles)

        # 4. Final half-step velocity update
        updated = []
        for p, f in zip(half_step_particles, f_new):
            acc_new = f / p.mass
            final_vel = p.velocity + (0.5 * dt * acc_new)
            updated.append(p.with_velocity(final_vel).with_force(f))

        return updated


class RK4Integrator(BaseIntegrator):
    """Classic 4th-Order Runge-Kutta Integrator for High Accuracy ODE Integration."""

    def step(self, particles: List[Particle], dt: float, force_field: ForceField) -> List[Particle]:
        n = len(particles)

        # Helper to compute accelerations from a list of positions
        def get_acc(pos_list: List[Vector2D], vel_list: List[Vector2D]) -> List[Vector2D]:
            eval_particles = [
                p.with_position(pos).with_velocity(vel)
                for p, pos, vel in zip(particles, pos_list, vel_list)
            ]
            forces = force_field.compute_net_forces(eval_particles)
            return [f / p.mass for p, f in zip(particles, forces)]

        x0 = [p.position for p in particles]
        v0 = [p.velocity for p in particles]

        # k1
        a0 = get_acc(x0, v0)
        k1_x = v0
        k1_v = a0

        # k2
        x1 = [x + 0.5 * dt * kx for x, kx in zip(x0, k1_x)]
        v1 = [v + 0.5 * dt * kv for v, kv in zip(v0, k1_v)]
        a1 = get_acc(x1, v1)
        k2_x = v1
        k2_v = a1

        # k3
        x2 = [x + 0.5 * dt * kx for x, kx in zip(x0, k2_x)]
        v2 = [v + 0.5 * dt * kv for v, kv in zip(v0, k2_v)]
        a2 = get_acc(x2, v2)
        k3_x = v2
        k3_v = a2

        # k4
        x3 = [x + dt * kx for x, kx in zip(x0, k3_x)]
        v3 = [v + dt * kv for v, kv in zip(v0, k3_v)]
        a3 = get_acc(x3, v3)
        k4_x = v3
        k4_v = a3

        # Combine k1..k4
        updated = []
        for i in range(n):
            new_pos = x0[i] + (dt / 6.0) * (k1_x[i] + 2.0 * k2_x[i] + 2.0 * k3_x[i] + k4_x[i])
            new_vel = v0[i] + (dt / 6.0) * (k1_v[i] + 2.0 * k2_v[i] + 2.0 * k3_v[i] + k4_v[i])
            updated.append(particles[i].with_position(new_pos).with_velocity(new_vel))

        return updated
