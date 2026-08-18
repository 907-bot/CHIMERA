"""Unit Tests for Physics Integrators, Force Fields, and Collisions"""

import pytest
from packages.core.models import Vector2D, Particle, Boundary
from packages.physics.forces import GravityForce, HarmonicForce, DragForce, ForceField
from packages.physics.collisions import BoundaryCollision, ParticleCollision
from packages.physics.integrators import VerletIntegrator, RK4Integrator, EulerIntegrator


def test_gravity_force_computation():
    p1 = Particle(id=1, mass=1.0, position=Vector2D(x=0.0, y=0.0))
    p2 = Particle(id=2, mass=1.0, position=Vector2D(x=10.0, y=0.0))

    g_force = GravityForce(G=1.0, softening=0.0)
    forces = g_force.compute_forces([p1, p2])

    # F12 = G * m1 * m2 / r^2 = 1.0 / 100.0 = 0.01 in +x direction for p1
    assert forces[0].x == pytest.approx(0.01, abs=1e-5)
    assert forces[0].y == pytest.approx(0.0, abs=1e-5)
    # Equal and opposite for p2
    assert forces[1].x == pytest.approx(-0.01, abs=1e-5)
    assert forces[1].y == pytest.approx(0.0, abs=1e-5)


def test_boundary_collision_reflection():
    boundary = Boundary(x_min=0.0, x_max=100.0, y_min=0.0, y_max=100.0)
    # Particle heading left out of bounds
    p = Particle(id=1, radius=1.0, position=Vector2D(x=0.5, y=50.0), velocity=Vector2D(x=-5.0, y=0.0))

    handler = BoundaryCollision(restitution=1.0)
    resolved = handler.resolve([p], boundary)

    assert resolved[0].position.x == 1.0  # Clamped to x_min + radius
    assert resolved[0].velocity.x == 5.0  # Reversed velocity


def test_particle_elastic_collision():
    # Two identical particles moving towards each other on X-axis
    p1 = Particle(id=1, mass=1.0, radius=1.0, position=Vector2D(x=10.0, y=50.0), velocity=Vector2D(x=2.0, y=0.0))
    p2 = Particle(id=2, mass=1.0, radius=1.0, position=Vector2D(x=11.5, y=50.0), velocity=Vector2D(x=-2.0, y=0.0))

    handler = ParticleCollision(restitution=1.0)
    resolved = handler.resolve([p1, p2])

    # Velocities should swap in perfectly elastic collision between equal masses
    assert resolved[0].velocity.x == pytest.approx(-2.0, abs=1e-4)
    assert resolved[1].velocity.x == pytest.approx(2.0, abs=1e-4)


def test_integrator_stepping():
    p1 = Particle(id=1, mass=1.0, position=Vector2D(x=10.0, y=0.0), velocity=Vector2D(x=0.0, y=1.0))
    p2 = Particle(id=2, mass=1.0, position=Vector2D(x=-10.0, y=0.0), velocity=Vector2D(x=0.0, y=-1.0))

    force_field = ForceField([GravityForce(G=1.0, softening=0.1)])

    verlet = VerletIntegrator()
    rk4 = RK4Integrator()
    euler = EulerIntegrator()

    step_v = verlet.step([p1, p2], dt=0.01, force_field=force_field)
    step_r = rk4.step([p1, p2], dt=0.01, force_field=force_field)
    step_e = euler.step([p1, p2], dt=0.01, force_field=force_field)

    assert len(step_v) == 2
    assert len(step_r) == 2
    assert len(step_e) == 2
