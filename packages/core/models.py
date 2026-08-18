"""Immutable Core Data Models for CHIMERA Physics Engine"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class Vector2D(BaseModel):
    """Immutable 2D vector representation with vector arithmetic."""
    model_config = ConfigDict(frozen=True)

    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: Vector2D) -> Vector2D:
        return Vector2D(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: Vector2D) -> Vector2D:
        return Vector2D(x=self.x - other.x, y=self.y - other.y)

    def __mul__(self, scalar: float) -> Vector2D:
        return Vector2D(x=self.x * scalar, y=self.y * scalar)

    def __rmul__(self, scalar: float) -> Vector2D:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vector2D:
        if scalar == 0:
            raise ZeroDivisionError("Vector division by zero")
        return Vector2D(x=self.x / scalar, y=self.y / scalar)

    def __neg__(self) -> Vector2D:
        return Vector2D(x=-self.x, y=-self.y)

    def dot(self, other: Vector2D) -> float:
        """Compute scalar dot product with another vector."""
        return self.x * other.x + self.y * other.y

    def norm_sq(self) -> float:
        """Square of Euclidean norm."""
        return self.x * self.x + self.y * self.y

    def norm(self) -> float:
        """Euclidean norm (magnitude)."""
        return math.sqrt(self.norm_sq())

    def distance(self, other: Vector2D) -> float:
        """Euclidean distance to another vector."""
        return (self - other).norm()

    def normalize(self) -> Vector2D:
        """Unit vector in the direction of this vector."""
        n = self.norm()
        if n == 0:
            return Vector2D(x=0.0, y=0.0)
        return self / n

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


class Particle(BaseModel):
    """Immutable physical particle in 2D space."""
    model_config = ConfigDict(frozen=True)

    id: int
    mass: float = Field(gt=0.0, default=1.0)
    radius: float = Field(gt=0.0, default=1.0)
    position: Vector2D
    velocity: Vector2D = Vector2D(x=0.0, y=0.0)
    force: Vector2D = Vector2D(x=0.0, y=0.0)

    def with_position(self, new_pos: Vector2D) -> Particle:
        return self.model_copy(update={"position": new_pos})

    def with_velocity(self, new_vel: Vector2D) -> Particle:
        return self.model_copy(update={"velocity": new_vel})

    def with_force(self, new_force: Vector2D) -> Particle:
        return self.model_copy(update={"force": new_force})


class Boundary(BaseModel):
    """2D rectangular boundary box."""
    model_config = ConfigDict(frozen=True)

    x_min: float = 0.0
    x_max: float = 100.0
    y_min: float = 0.0
    y_max: float = 100.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


class WorldConfig(BaseModel):
    """Configuration specification for a simulation world run."""
    model_config = ConfigDict(frozen=True)

    world_id: str = "world_001"
    boundary: Boundary = Boundary()
    num_particles: int = Field(gt=0, default=10)
    dt: float = Field(gt=0.0, default=0.01)
    seed: int = 42
    integrator_type: str = "verlet"  # 'verlet', 'rk4', 'euler'
    gravity_constant: float = 1.0
    softening: float = 0.1
    restitution: float = 1.0  # Coefficient of restitution (1.0 = perfectly elastic)


class WorldState(BaseModel):
    """Immutable snapshot of the universe state at a single step/tick."""
    model_config = ConfigDict(frozen=True)

    world_id: str
    step: int = Field(ge=0, default=0)
    time: float = Field(ge=0.0, default=0.0)
    dt: float = Field(gt=0.0, default=0.01)
    particles: List[Particle]
    boundary: Boundary
    seed: int
    config_hash: str = ""

    def get_particle_by_id(self, particle_id: int) -> Optional[Particle]:
        for p in self.particles:
            if p.id == particle_id:
                return p
        return None
