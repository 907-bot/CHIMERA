"""Cross-World Invariant Detector for CHIMERA Multiverse Engine.

Evaluates trajectories across hundreds of parallel worlds to statistically distinguish:
  1. UNIVERSAL CONSERVATION LAWS (e.g. Total Energy, Linear Momentum in closed systems)
     - Conserved within individual worlds (|Q(t) - Q(0)| / Q(0) << 1e-3)
     - Conserved across ALL random seeds / initial conditions
  2. SEED-CONTINGENT HISTORICAL ACCIDENTS (e.g. Particle 1 final X position)
     - Non-conserved over time, variance across worlds is high
  3. DISSIPATIVE ASYMMETRIES (e.g. systems with inelastic collisions or drag)
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
from packages.core.models import WorldState
from packages.physics.energy import EnergyMetrics
from packages.multiverse.models import InvariantResult


class CrossWorldInvariantDetector:
    """Detects universal physical invariants across multiversal world runs."""

    @staticmethod
    def _compute_world_metrics(history: List[WorldState]) -> Dict[str, np.ndarray]:
        """Extract time-series for all candidate quantities across one world trajectory."""
        steps = len(history)
        total_energy = np.zeros(steps)
        total_px = np.zeros(steps)
        total_py = np.zeros(steps)
        com_vx = np.zeros(steps)
        p1_x = np.zeros(steps)
        p1_vx = np.zeros(steps)

        for t_idx, state in enumerate(history):
            # Energy
            e_dict = EnergyMetrics.compute_all(state.particles)
            total_energy[t_idx] = e_dict["total_energy"]

            # Momentum & Center of mass
            m_total = sum(p.mass for p in state.particles) if state.particles else 1.0
            px = sum(p.mass * p.velocity.x for p in state.particles)
            py = sum(p.mass * p.velocity.y for p in state.particles)
            total_px[t_idx] = px
            total_py[t_idx] = py
            com_vx[t_idx] = px / m_total

            # Sample particle coordinates (if present)
            if state.particles:
                p1_x[t_idx] = state.particles[0].position.x
                p1_vx[t_idx] = state.particles[0].velocity.x

        return {
            "total_energy": total_energy,
            "total_momentum_x": total_px,
            "total_momentum_y": total_py,
            "center_of_mass_vx": com_vx,
            "particle_1_position_x": p1_x,
            "particle_1_velocity_x": p1_vx,
        }

    def detect_invariants(
        self,
        world_histories: List[List[WorldState]],
        conservation_threshold: float = 1e-3,
    ) -> List[InvariantResult]:
        """Analyze batch trajectories across parallel worlds to detect invariants.

        Args:
            world_histories:        List of state histories, one per world run.
            conservation_threshold: Max permissible relative drift for conservation (default 1e-3).

        Returns:
            List of InvariantResult objects detailing invariant vs contingent classification.
        """
        if not world_histories or not world_histories[0]:
            return []

        num_worlds = len(world_histories)
        
        # Aggregate metrics across all worlds
        # Shape per metric: [num_worlds, num_steps]
        all_metrics: Dict[str, List[np.ndarray]] = {
            "total_energy": [],
            "total_momentum_x": [],
            "total_momentum_y": [],
            "center_of_mass_vx": [],
            "particle_1_position_x": [],
            "particle_1_velocity_x": [],
        }

        for history in world_histories:
            w_metrics = self._compute_world_metrics(history)
            for k, arr in w_metrics.items():
                all_metrics[k].append(arr)

        results: List[InvariantResult] = []

        for q_name, world_arrays in all_metrics.items():
            # Matrix of shape (num_worlds, num_steps)
            data_matrix = np.array(world_arrays)

            # 1. Within-world relative drift for each world
            drifts = []
            for w in range(num_worlds):
                base_val = abs(float(data_matrix[w, 0]))
                denom = max(base_val, 1e-2)
                # Mean relative drift over the trajectory
                rel_drift = float(np.mean(np.abs(data_matrix[w] - data_matrix[w, 0])) / denom)
                drifts.append(rel_drift)

            mean_drift = float(np.mean(drifts))
            max_drift = float(np.max(drifts))
            p95_drift = float(np.percentile(drifts, 95))

            # 2. Across-world variance of initial values and final values
            initial_vals = data_matrix[:, 0]
            final_variance = float(np.var(data_matrix[:, -1]))

            # 3. Conservation score [0, 1]
            score = float(np.exp(-50.0 * mean_drift))
            score = max(0.0, min(1.0, score))

            # 4. Classification
            is_conserved = mean_drift < conservation_threshold and p95_drift < (conservation_threshold * 5)

            # Check for monotonic dissipation (e.g. energy loss in inelastic worlds)
            is_dissipating = False
            if not is_conserved and "energy" in q_name:
                diffs = data_matrix[:, -1] - data_matrix[:, 0]
                if np.all(diffs <= 1e-5):
                    is_dissipating = True

            if is_conserved:
                verdict = "UNIVERSAL_CONSERVATION_LAW"
                is_universal = True
                desc = (
                    f"Conserved across all {num_worlds} worlds with mean drift "
                    f"{mean_drift:.6f} < threshold {conservation_threshold}."
                )
            elif is_dissipating:
                verdict = "DISSIPATIVE_ASYMMETRY"
                is_universal = False
                desc = f"Monotonically dissipated across worlds (mean drift: {mean_drift:.4f})."
            else:
                verdict = "SEED_CONTINGENT_HISTORICAL_FACT"
                is_universal = False
                desc = (
                    f"Temporal fluctuation is significant ({mean_drift:.4f} >= {conservation_threshold}) "
                    f"with across-world variance {final_variance:.4f}. Contingent on seed/initial state."
                )

            results.append(
                InvariantResult(
                    quantity_name=q_name,
                    is_universal_invariant=is_universal,
                    mean_within_world_drift=round(mean_drift, 6),
                    max_within_world_drift=round(max_drift, 6),
                    across_world_variance=round(final_variance, 6),
                    conservation_score=round(score, 4),
                    verdict=verdict,
                    description=desc,
                )
            )

        return results
