"""3D Gravitational N-Body Cosmological Integrator (CHIMERA v5.0 - Phase 13)

Integrates orbital mechanics using Symplectic Velocity-Verlet in 3D:
    r_{t+dt} = r_t + v_t dt + 0.5 a_t dt²
    a_{t+dt} = - sum_j G m_j (r_i - r_j) / (|r_i - r_j|² + ε²)^(3/2)
    v_{t+dt} = v_t + 0.5 (a_t + a_{t+dt}) dt
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
from packages.cosmology.models import CelestialBody, CosmologicalWorldConfig


class NBodyCosmologyEngine:
    """3D symplectic gravitational integrator for multi-planetary and galactic systems."""

    def __init__(self, config: CosmologicalWorldConfig):
        self.config = config
        self.G = config.g_grav
        self.eps = config.softening
        self.dt = config.dt

    def _compute_accelerations(self, positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """Compute gravitational accelerations for all N bodies vectorially."""
        n = len(masses)
        # diff[i, j, :] = r[j] - r[i]
        diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        dist_sq = np.sum(diff ** 2, axis=-1) + self.eps ** 2
        inv_dist_cube = dist_sq ** (-1.5)

        # Set diagonal to 0 (no self-gravity)
        np.fill_diagonal(inv_dist_cube, 0.0)

        # Force weights: mass_j * inv_dist_cube[i, j]
        weights = masses[np.newaxis, :] * inv_dist_cube  # (n, n)
        # acc[i] = G * sum_j weights[i, j] * diff[i, j]
        acc = self.G * np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        return acc

    def step(self, bodies: List[CelestialBody]) -> List[CelestialBody]:
        """Advance all celestial bodies by one symplectic Velocity-Verlet step."""
        n = len(bodies)
        positions = np.array([b.position for b in bodies], dtype=np.float64)
        velocities = np.array([b.velocity for b in bodies], dtype=np.float64)
        masses = np.array([b.mass for b in bodies], dtype=np.float64)

        # Initial acceleration a(t)
        a_curr = self._compute_accelerations(positions, masses)

        # Update position: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        pos_next = positions + velocities * self.dt + 0.5 * a_curr * (self.dt ** 2)

        # Compute new acceleration a(t+dt)
        a_next = self._compute_accelerations(pos_next, masses)

        # Update velocity: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        vel_next = velocities + 0.5 * (a_curr + a_next) * self.dt

        new_bodies = []
        for i, b in enumerate(bodies):
            new_bodies.append(
                b.model_copy(
                    update={
                        "position": (float(pos_next[i, 0]), float(pos_next[i, 1]), float(pos_next[i, 2])),
                        "velocity": (float(vel_next[i, 0]), float(vel_next[i, 1]), float(vel_next[i, 2])),
                    }
                )
            )
        return new_bodies

    def compute_energy_and_momentum(self, bodies: List[CelestialBody]) -> Dict[str, float]:
        """Calculates total kinetic energy, potential energy, and angular momentum."""
        pos = np.array([b.position for b in bodies], dtype=np.float64)
        vel = np.array([b.velocity for b in bodies], dtype=np.float64)
        masses = np.array([b.mass for b in bodies], dtype=np.float64)

        # Kinetic energy = 0.5 * sum(m * v^2)
        ke = 0.5 * np.sum(masses * np.sum(vel ** 2, axis=1))

        # Potential energy = - G * sum_{i < j} (m_i * m_j / r_ij)
        n = len(bodies)
        pe = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(pos[i] - pos[j]) + self.eps
                pe -= self.G * masses[i] * masses[j] / r

        # Angular momentum L = sum(m * r x v)
        ang_mom = np.sum(masses[:, np.newaxis] * np.cross(pos, vel), axis=0)
        total_l = float(np.linalg.norm(ang_mom))

        return {
            "kinetic_energy": float(ke),
            "potential_energy": float(pe),
            "total_energy": float(ke + pe),
            "angular_momentum_magnitude": total_l,
        }
