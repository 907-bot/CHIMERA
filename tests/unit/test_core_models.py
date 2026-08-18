"""Unit Tests for Core Models, Vector Math, and Serialization"""

import pytest
from packages.core.models import Vector2D, Particle, Boundary, WorldState, WorldConfig
from packages.core.serialization import (
    hash_world_state,
    world_state_to_dict,
    world_state_from_dict,
)


def test_vector_operations():
    v1 = Vector2D(x=3.0, y=4.0)
    v2 = Vector2D(x=1.0, y=2.0)

    # Norm & Norm_sq
    assert v1.norm_sq() == 25.0
    assert v1.norm() == 5.0

    # Addition & Subtraction
    assert (v1 + v2) == Vector2D(x=4.0, y=6.0)
    assert (v1 - v2) == Vector2D(x=2.0, y=2.0)

    # Multiplication & Division
    assert (v1 * 2.0) == Vector2D(x=6.0, y=8.0)
    assert (2.0 * v1) == Vector2D(x=6.0, y=8.0)
    assert (v1 / 2.0) == Vector2D(x=1.5, y=2.0)

    # Dot product & Distance
    assert v1.dot(v2) == 11.0
    assert v1.distance(v2) == pytest.approx(2.828427, abs=1e-5)

    # Normalize
    n = v1.normalize()
    assert n.norm() == pytest.approx(1.0, abs=1e-6)
    assert n.x == pytest.approx(0.6, abs=1e-6)
    assert n.y == pytest.approx(0.8, abs=1e-6)


def test_vector_division_by_zero():
    v = Vector2D(x=1.0, y=1.0)
    with pytest.raises(ZeroDivisionError):
        _ = v / 0.0


def test_particle_immutability():
    p = Particle(id=1, mass=2.0, radius=0.5, position=Vector2D(x=10.0, y=10.0))
    with pytest.raises((TypeError, Exception)):
        p.mass = 5.0  # Pydantic frozen model


def test_serialization_roundtrip():
    boundary = Boundary(x_min=0.0, x_max=100.0, y_min=0.0, y_max=100.0)
    p1 = Particle(id=1, mass=1.0, position=Vector2D(x=10.0, y=20.0), velocity=Vector2D(x=1.0, y=-1.0))
    p2 = Particle(id=2, mass=2.0, position=Vector2D(x=30.0, y=40.0), velocity=Vector2D(x=-0.5, y=0.5))

    state = WorldState(
        world_id="test_world",
        step=10,
        time=0.1,
        dt=0.01,
        particles=[p1, p2],
        boundary=boundary,
        seed=123,
    )

    state_dict = world_state_to_dict(state)
    restored_state = world_state_from_dict(state_dict)

    assert restored_state == state
    assert hash_world_state(restored_state) == hash_world_state(state)
