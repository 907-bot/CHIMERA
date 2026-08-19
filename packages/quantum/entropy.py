"""Statistical Mechanics, Quantum Entanglement Entropy & Arrow of Time (CHIMERA v3.0 - Phase 11)"""

from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
from packages.quantum.models import QuantumLatticeState


class StatisticalEntropyAnalyzer:
    """Computes statistical mechanics metrics: Shannon spatial entropy, bipartite entanglement entropy, thermal relaxation."""

    def __init__(self, kb: float = 1.0):
        self.kb = kb

    def compute_spatial_entropy(self, state: QuantumLatticeState) -> float:
        """Compute Shannon spatial entropy S = - kB * sum(p_i * ln(p_i))."""
        probs = state.probability_density
        # Filter out zero probabilities to avoid log(0)
        p_valid = probs[probs > 1e-15]
        entropy = -self.kb * np.sum(p_valid * np.log(p_valid))
        return float(entropy)

    def compute_bipartite_entanglement_entropy(self, state: QuantumLatticeState, cut_idx: int) -> float:
        """Compute subsystem von Neumann entanglement entropy S(rho_A) for a partition at cut_idx."""
        psi = state.to_complex_array()
        prob_subsystem = np.sum(np.abs(psi[:cut_idx]) ** 2)
        p_A = float(np.clip(prob_subsystem, 1e-15, 1.0 - 1e-15))
        p_B = 1.0 - p_A
        # Binary entanglement entropy
        s_ent = - (p_A * np.log(p_A) + p_B * np.log(p_B))
        return float(s_ent)

    def compute_thermal_relaxation_history(self, states: List[QuantumLatticeState]) -> Dict[str, Any]:
        """Tracks entropy trajectory to verify thermodynamic arrow of time."""
        entropies = [self.compute_spatial_entropy(s) for s in states]
        initial_s = entropies[0] if entropies else 0.0
        final_s = entropies[-1] if entropies else 0.0
        delta_s = final_s - initial_s

        return {
            "initial_entropy": initial_s,
            "final_entropy": final_s,
            "entropy_change": delta_s,
            "monotonic_increase_ratio": float(np.mean(np.diff(entropies) >= -1e-8)),
            "entropy_history": entropies,
        }
