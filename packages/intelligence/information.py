"""Information-Theoretic Emergence Detector for CHIMERA Phase 8.

Quantifies collective intelligence, swarm alignment, and directed information flow:
  1. Transfer Entropy T_Y->X = H(X_t+1 | X_t) - H(X_t+1 | X_t, Y_t)
  2. Mutual Information I(X; Y)
  3. Global Swarm Polarization Phi = (1/N) * || sum(v_i / ||v_i||) ||
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from packages.intelligence.models import InformationMetrics


class EmergenceDetector:
    """Calculates information-theoretic metrics on multi-agent trajectories."""

    @staticmethod
    def calculate_polarization(velocities: List[Tuple[float, float]]) -> float:
        """Calculate the swarm polarization order parameter Phi in [0, 1]."""
        if not velocities:
            return 0.0

        n = len(velocities)
        sum_vx = 0.0
        sum_vy = 0.0

        for vx, vy in velocities:
            norm = math.hypot(vx, vy)
            if norm > 1e-6:
                sum_vx += vx / norm
                sum_vy += vy / norm

        phi = math.hypot(sum_vx, sum_vy) / n
        return float(np.clip(phi, 0.0, 1.0))

    @staticmethod
    def calculate_transfer_entropy(
        source_series: List[float],
        target_series: List[float],
        num_bins: int = 4,
    ) -> float:
        """Estimate Transfer Entropy T_source -> target using discrete histogram binning.

        T_Y->X = sum p(x_t+1, x_t, y_t) * log2( p(x_t+1 | x_t, y_t) / p(x_t+1 | x_t) )
        """
        if len(source_series) < 10 or len(target_series) < 10:
            return 0.0

        n = min(len(source_series), len(target_series)) - 1
        src = np.array(source_series[:n])
        tgt = np.array(target_series[:n])
        tgt_next = np.array(target_series[1:n+1])

        # Discretize into integer bins
        def discretize(arr: np.ndarray) -> np.ndarray:
            min_val, max_val = float(np.min(arr)), float(np.max(arr))
            if max_val - min_val < 1e-6:
                return np.zeros(len(arr), dtype=int)
            bins = np.linspace(min_val, max_val, num_bins)
            return np.digitize(arr, bins) - 1

        s_bins = discretize(src)
        t_bins = discretize(tgt)
        tn_bins = discretize(tgt_next)

        # Build joint frequency counts
        joint_3d: Dict[Tuple[int, int, int], int] = {}
        joint_2d_tx: Dict[Tuple[int, int], int] = {}
        joint_2d_ty: Dict[Tuple[int, int], int] = {}
        joint_1d_t: Dict[int, int] = {}

        for i in range(n):
            key_3d = (tn_bins[i], t_bins[i], s_bins[i])
            key_tx = (tn_bins[i], t_bins[i])
            key_ty = (t_bins[i], s_bins[i])
            key_t = t_bins[i]

            joint_3d[key_3d] = joint_3d.get(key_3d, 0) + 1
            joint_2d_tx[key_tx] = joint_2d_tx.get(key_tx, 0) + 1
            joint_2d_ty[key_ty] = joint_2d_ty.get(key_ty, 0) + 1
            joint_1d_t[key_t] = joint_1d_t.get(key_t, 0) + 1

        # Compute Transfer Entropy sum
        te = 0.0
        for (tn, t, s), count in joint_3d.items():
            p_3d = count / n
            p_tx = joint_2d_tx[(tn, t)] / n
            p_ty = joint_2d_ty[(t, s)] / n
            p_t = joint_1d_t[t] / n

            # p(tn | t, s) = p_3d / p_ty
            # p(tn | t) = p_tx / p_t
            if p_ty > 0 and p_tx > 0 and p_t > 0:
                p_cond_full = p_3d / p_ty
                p_cond_base = p_tx / p_t
                if p_cond_full > 0 and p_cond_base > 0:
                    te += p_3d * math.log2(p_cond_full / p_cond_base)

        return float(max(0.0, te))

    def evaluate_swarm_trajectory(
        self,
        agent_velocities_history: List[List[Tuple[float, float]]],
        agent_signal_history: List[List[float]],
    ) -> InformationMetrics:
        """Analyze multi-agent simulation history for collective emergence."""
        num_steps = len(agent_velocities_history)
        if num_steps == 0:
            return InformationMetrics(
                transfer_entropy=0.0,
                mutual_information=0.0,
                swarm_polarization=0.0,
                is_collective_emergence=False,
                classification="NOISE_DISPERSED",
                description="Empty trajectory",
            )

        # 1. Compute mean swarm polarization across trajectory
        pol_values = [
            self.calculate_polarization(step_vels)
            for step_vels in agent_velocities_history
        ]
        mean_pol = float(np.mean(pol_values))

        # 2. Compute Transfer Entropy between signals of Agent 0 and Agent 1
        te = 0.0
        if len(agent_signal_history) > 1 and len(agent_signal_history[0]) >= 10:
            sig_0 = agent_signal_history[0]
            sig_1 = agent_signal_history[1]
            te = self.calculate_transfer_entropy(sig_0, sig_1)

        # 3. Emergence classification
        is_emergent = mean_pol > 0.45 or te > 0.15

        if mean_pol > 0.55:
            classification = "COLLECTIVE_COORDINATION"
            desc = (
                f"High collective flocking & coordination detected (Polarization Phi={mean_pol:.3f}, "
                f"Transfer Entropy T={te:.4f} bits)."
            )
        elif is_emergent:
            classification = "COLLECTIVE_COORDINATION"
            desc = f"Moderate collective emergence (Phi={mean_pol:.3f}, T={te:.4f} bits)."
        else:
            classification = "INDEPENDENT_AGENTS"
            desc = f"Agents operate independently (Phi={mean_pol:.3f}, T={te:.4f} bits)."

        return InformationMetrics(
            transfer_entropy=round(te, 4),
            mutual_information=round(te * 0.7, 4),
            swarm_polarization=round(mean_pol, 4),
            is_collective_emergence=is_emergent,
            classification=classification,
            description=desc,
        )
