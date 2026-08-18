"""Deterministic Force Field Computations for CHIMERA Engine"""

from abc import ABC, abstractmethod
from typing import List
import math
import numpy as np
from packages.core.models import Particle, Vector2D


class BaseForce(ABC):
    """Abstract Base Class for physical force implementations."""
    @abstractmethod
    def compute_forces(self, particles: List[Particle]) -> List[Vector2D]:
        """Compute net force vector for each particle in the system."""
        pass


class GravityForce(BaseForce):
    """Pairwise N-body Gravitational Force with Softening Parameter."""
    def __init__(self, G: float = 1.0, softening: float = 0.1):
        self.G = G
        self.softening = softening

    def compute_forces(self, particles: List[Particle]) -> List[Vector2D]:
        n = len(particles)
        forces = [Vector2D(x=0.0, y=0.0) for _ in range(n)]

        if n < 2:
            return forces

        # Vectorized NumPy pairwise calculation
        pos = np.array([[p.position.x, p.position.y] for p in particles], dtype=np.float64)
        masses = np.array([p.mass for p in particles], dtype=np.float64)

        # Difference matrix delta[i, j] = pos[j] - pos[i]
        delta = pos[None, :, :] - pos[:, None, :]
        dist_sq = np.sum(delta ** 2, axis=-1) + self.softening ** 2
        inv_dist_cube = dist_sq ** (-1.5)
        np.fill_diagonal(inv_dist_cube, 0.0)

        # Force magnitude array F_matrix[i, j] = G * m[i] * m[j] / (dist^2 + eps^2)^(3/2)
        f_matrix = self.G * (masses[:, None] * masses[None, :]) * inv_dist_cube

        # Net force on particle i: sum_j F_matrix[i, j] * delta[i, j]
        force_vecs = np.sum(f_matrix[:, :, None] * delta, axis=1)

        return [Vector2D(x=float(f[0]), y=float(f[1])) for f in force_vecs]


class HarmonicForce(BaseForce):
    """Central Hooke's Law Spring Force F = -k * (r - center)."""
    def __init__(self, k: float = 1.0, center: Vector2D = Vector2D(x=50.0, y=50.0)):
        self.k = k
        self.center = center

    def compute_forces(self, particles: List[Particle]) -> List[Vector2D]:
        forces = []
        for p in particles:
            disp = p.position - self.center
            f = -self.k * disp
            forces.append(f)
        return forces


class DragForce(BaseForce):
    """Velocity Damping Force F = -gamma * v."""
    def __init__(self, gamma: float = 0.01):
        self.gamma = gamma

    def compute_forces(self, particles: List[Particle]) -> List[Vector2D]:
        return [-self.gamma * p.velocity for p in particles]


class ForceField:
    """Aggregator that sums all active forces on each particle."""
    def __init__(self, forces: List[BaseForce] = None):
        self.forces = forces if forces is not None else []

    def add_force(self, force: BaseForce):
        self.forces.append(force)

    def compute_net_forces(self, particles: List[Particle]) -> List[Vector2D]:
        n = len(particles)
        net_forces = [Vector2D(x=0.0, y=0.0) for _ in range(n)]

        for force_law in self.forces:
            f_list = force_law.compute_forces(particles)
            for i in range(n):
                net_forces[i] = net_forces[i] + f_list[i]

        return net_forces
