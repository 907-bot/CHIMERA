"""Derived Observable Feature Extraction and Blind Observation Sanitizer"""

from typing import List, Dict, Any, Tuple
import math
import numpy as np
from pydantic import BaseModel, ConfigDict
from packages.core.models import WorldState, Particle, Vector2D, Boundary


class BlindObservation(BaseModel):
    """Sanitized Observation Snapshot provided to AI Scientists (Withholding Hidden Laws)."""
    model_config = ConfigDict(frozen=True)

    world_id: str
    step: int
    time: float
    dt: float
    particles_positions: List[Tuple[float, float]]
    particles_velocities: List[Tuple[float, float]]
    spatial_entropy: float
    kinetic_energy: float
    total_momentum: Tuple[float, float]


class FeatureExtractor:
    """Calculates Observable Spatial, Kinetic, and Statistical Mechanics Metrics."""

    @staticmethod
    def spatial_entropy(
        particles: List[Particle],
        grid_size: int = 10,
        boundary: Boundary = Boundary()
    ) -> float:
        """Compute Spatial Occupancy Shannon Entropy S = - sum(p_i * ln(p_i))."""
        if not particles:
            return 0.0

        num_cells = grid_size * grid_size
        counts = np.zeros(num_cells, dtype=np.int32)

        dx = boundary.width / grid_size
        dy = boundary.height / grid_size

        for p in particles:
            ix = int((p.position.x - boundary.x_min) / dx)
            iy = int((p.position.y - boundary.y_min) / dy)

            # Clamp indices
            ix = max(0, min(grid_size - 1, ix))
            iy = max(0, min(grid_size - 1, iy))

            cell_idx = iy * grid_size + ix
            counts[cell_idx] += 1

        probs = counts / len(particles)
        probs = probs[probs > 0]  # Filter out empty cells

        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)

    @staticmethod
    def pair_correlation(
        particles: List[Particle],
        r_max: float = 50.0,
        num_bins: int = 50
    ) -> Tuple[List[float], List[float]]:
        """Compute Radial Distribution Function g(r)."""
        n = len(particles)
        if n < 2:
            return [], []

        dr = r_max / num_bins
        bin_edges = np.linspace(0, r_max, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        hist = np.zeros(num_bins, dtype=np.int32)

        pos = np.array([[p.position.x, p.position.y] for p in particles])
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < r_max:
                    bin_idx = int(d / dr)
                    if bin_idx < num_bins:
                        hist[bin_idx] += 2

        # Normalization factor for 2D density
        area = math.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
        density = n / (100.0 * 100.0)  # Assuming standard 100x100 box
        g_r = hist / (n * density * area)

        return bin_centers.tolist(), g_r.tolist()

    @staticmethod
    def mean_squared_displacement(
        trajectory: List[WorldState],
        particle_id: int = 1
    ) -> List[Tuple[float, float]]:
        """Compute Mean Squared Displacement MSD(t) = |r(t) - r(0)|^2 over time."""
        if not trajectory:
            return []

        initial_p = trajectory[0].get_particle_by_id(particle_id)
        if not initial_p:
            return []

        r0 = initial_p.position
        msd_list = []

        for state in trajectory:
            p = state.get_particle_by_id(particle_id)
            if p:
                disp_sq = (p.position - r0).norm_sq()
                msd_list.append((state.time, disp_sq))

        return msd_list


class ObservationMask:
    """Sanitizer enforcing Rule 1: Withholding underlying physical force equations."""

    @staticmethod
    def mask_state(state: WorldState) -> BlindObservation:
        """Filter raw WorldState into a BlindObservation object stripped of hidden constants."""
        positions = [(p.position.x, p.position.y) for p in state.particles]
        velocities = [(p.velocity.x, p.velocity.y) for p in state.particles]

        entropy = FeatureExtractor.spatial_entropy(state.particles, boundary=state.boundary)

        ke = sum(0.5 * p.mass * p.velocity.norm_sq() for p in state.particles)
        total_px = sum(p.mass * p.velocity.x for p in state.particles)
        total_py = sum(p.mass * p.velocity.y for p in state.particles)

        return BlindObservation(
            world_id=state.world_id,
            step=state.step,
            time=state.time,
            dt=state.dt,
            particles_positions=positions,
            particles_velocities=velocities,
            spatial_entropy=entropy,
            kinetic_energy=ke,
            total_momentum=(total_px, total_py),
        )
