"""Energy and Conservation Law Computation Metrics"""

from typing import List, Dict, Any
import math
from packages.core.models import Particle, Vector2D


class EnergyMetrics:
    """Calculates Kinetic, Potential, and Total System Energy metrics."""

    @staticmethod
    def kinetic_energy(particles: List[Particle]) -> float:
        """Compute total kinetic energy E_k = 0.5 * sum(m * v^2)."""
        ke = 0.0
        for p in particles:
            ke += 0.5 * p.mass * p.velocity.norm_sq()
        return ke

    @staticmethod
    def gravitational_potential_energy(particles: List[Particle], G: float = 1.0, softening: float = 0.1) -> float:
        """Compute gravitational potential energy E_p = - sum(G * m1 * m2 / sqrt(r^2 + eps^2))."""
        pe = 0.0
        n = len(particles)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = particles[i], particles[j]
                r_sq = (p1.position - p2.position).norm_sq()
                dist = math.sqrt(r_sq + softening ** 2)
                pe -= (G * p1.mass * p2.mass) / dist
        return pe

    @staticmethod
    def harmonic_potential_energy(particles: List[Particle], k: float = 1.0, center: Vector2D = Vector2D(x=50.0, y=50.0)) -> float:
        """Compute harmonic spring potential energy E_p = 0.5 * sum(k * (r - center)^2)."""
        pe = 0.0
        for p in particles:
            disp_sq = (p.position - center).norm_sq()
            pe += 0.5 * k * disp_sq
        return pe

    @staticmethod
    def total_momentum(particles: List[Particle]) -> Vector2D:
        """Compute total linear momentum P = sum(m * v)."""
        p_total = Vector2D(x=0.0, y=0.0)
        for p in particles:
            p_total = p_total + (p.mass * p.velocity)
        return p_total

    @classmethod
    def compute_all(
        cls,
        particles: List[Particle],
        G: float = 1.0,
        softening: float = 0.1,
        force_type: str = "gravity"
    ) -> Dict[str, Any]:
        ke = cls.kinetic_energy(particles)
        if force_type == "harmonic":
            pe = cls.harmonic_potential_energy(particles)
        else:
            pe = cls.gravitational_potential_energy(particles, G=G, softening=softening)
        
        total_e = ke + pe
        momentum = cls.total_momentum(particles)

        return {
            "kinetic_energy": ke,
            "potential_energy": pe,
            "total_energy": total_e,
            "momentum_x": momentum.x,
            "momentum_y": momentum.y,
            "momentum_norm": momentum.norm(),
        }
