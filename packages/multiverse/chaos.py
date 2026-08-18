"""Lyapunov Chaos & Trajectory Divergence Analysis for CHIMERA Multiverse Engine.

Implements maximal Lyapunov exponent estimation via twin deterministic trajectories:
  1. Base trajectory Z_base(t) = (r_1, ..., r_N, v_1, ..., v_N)
  2. Perturbed trajectory Z_pert(t) with initial micro-perturbation delta_0 = epsilon
  3. Distance delta(t) = ||Z_pert(t) - Z_base(t)||_2
  4. Log-linear regression: ln(delta(t) / delta_0) ~ lambda * t
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from packages.core.models import WorldConfig, WorldState, Vector2D, Particle
from packages.physics.engine import DeterministicEngine
from packages.multiverse.models import LyapunovResult


class LyapunovCalculator:
    """Calculates maximal Lyapunov exponents to classify system chaos."""

    @staticmethod
    def _state_to_phase_vector(state: WorldState) -> np.ndarray:
        """Convert a WorldState into a flat 4N phase space vector."""
        coords = []
        for p in sorted(state.particles, key=lambda x: x.id):
            coords.extend([p.position.x, p.position.y, p.velocity.x, p.velocity.y])
        return np.array(coords, dtype=np.float64)

    def calculate_lyapunov(
        self,
        base_config: WorldConfig,
        steps: int = 300,
        epsilon: float = 1e-8,
        saturation_fraction: float = 0.5,
    ) -> LyapunovResult:
        """Compute the maximum Lyapunov exponent by tracking twin trajectory divergence.

        Args:
            base_config:          Base world configuration.
            steps:                Total simulation steps to run.
            epsilon:              Magnitude of initial phase-space displacement.
            saturation_fraction:  Fraction of trajectory to use for exponential fitting
                                  before bounded phase-space saturation occurs.

        Returns:
            LyapunovResult with estimated lambda, R^2, and classification.
        """
        # 1. Run base simulation
        base_engine = DeterministicEngine(base_config)
        base_history = base_engine.run(steps)

        # 2. Construct perturbed initial state
        init_state = base_history[0]
        perturbed_particles = []
        
        # Apply micro-perturbation along deterministic direction
        for i, p in enumerate(init_state.particles):
            # Deterministic perturbation per particle
            dx = epsilon if i == 0 else 0.0
            dy = epsilon if i == 0 else 0.0
            perturbed_particles.append(
                Particle(
                    id=p.id,
                    mass=p.mass,
                    radius=p.radius,
                    position=Vector2D(x=p.position.x + dx, y=p.position.y + dy),
                    velocity=Vector2D(x=p.velocity.x, y=p.velocity.y),
                )
            )

        perturbed_init_state = WorldState(
            world_id=f"{base_config.world_id}_pert",
            step=0,
            time=0.0,
            dt=base_config.dt,
            particles=perturbed_particles,
            boundary=base_config.boundary,
            seed=base_config.seed,
            config_hash=base_engine.config_hash,
        )

        # 3. Run perturbed simulation
        pert_engine = DeterministicEngine(base_config)
        pert_engine.restore_state(perturbed_init_state)
        pert_history = pert_engine.run(steps)

        # 4. Compute phase-space Euclidean distance at each step
        deltas: List[float] = []
        times: List[float] = []

        for s_base, s_pert in zip(base_history, pert_history):
            v_base = self._state_to_phase_vector(s_base)
            v_pert = self._state_to_phase_vector(s_pert)
            dist = float(np.linalg.norm(v_pert - v_base))
            deltas.append(dist)
            times.append(s_base.time)

        deltas_arr = np.array(deltas)
        times_arr = np.array(times)

        # 5. Fit exponential growth regime ln(delta(t)) ~ lambda * t + C
        # Avoid log(0) and clamp before saturation
        fit_len = max(5, int(len(deltas_arr) * saturation_fraction))
        fit_deltas = deltas_arr[1:fit_len]
        fit_times = times_arr[1:fit_len]

        # Valid non-zero distances
        valid_mask = fit_deltas > 1e-15
        if np.sum(valid_mask) < 3:
            lambda_exp = 0.0
            r2 = 1.0
        else:
            x_vals = fit_times[valid_mask]
            y_vals = np.log(fit_deltas[valid_mask])

            # Linear regression: y = lambda * x + b
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            lambda_exp = float(slope)

            # R^2 calculation
            y_pred = slope * x_vals + intercept
            ss_res = np.sum((y_vals - y_pred) ** 2)
            ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
            r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))
            r2 = max(0.0, min(1.0, r2))

        # Classification
        if lambda_exp > 0.05 and r2 > 0.5:
            classification = "CHAOTIC"
            is_chaotic = True
        elif lambda_exp < -0.05:
            classification = "NEUTRAL_DAMPED"
            is_chaotic = False
        else:
            classification = "REGULAR_PERIODIC"
            is_chaotic = False

        return LyapunovResult(
            base_world_id=base_config.world_id,
            perturbed_world_id=f"{base_config.world_id}_pert",
            epsilon=epsilon,
            lyapunov_exponent=round(lambda_exp, 5),
            is_chaotic=is_chaotic,
            r_squared_fit=round(r2, 4),
            divergence_history=deltas,
            time_steps=times,
            classification=classification,
        )
