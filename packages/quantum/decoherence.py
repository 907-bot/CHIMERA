"""Many-Worlds Quantum Decoherence and Branching Manager (CHIMERA v3.0 - Phase 11)"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
from packages.quantum.models import QuantumLatticeState, BranchNode


class BranchingDecoherenceManager:
    """Manages Many-Worlds simulation branches upon projective measurements."""

    def __init__(self):
        self.branches: Dict[str, BranchNode] = {}

    def perform_spatial_measurement(
        self,
        parent_state: QuantumLatticeState,
        split_index: int,
    ) -> Tuple[BranchNode, BranchNode]:
        """Split wavefunction into two orthogonal measurement outcome branches:
        Branch L: particle detected in x < x[split_index]
        Branch R: particle detected in x >= x[split_index]
        """
        psi = parent_state.to_complex_array()
        N = len(psi)

        # Left projection operator
        psi_L = np.zeros_like(psi)
        psi_L[:split_index] = psi[:split_index]
        prob_L = float(np.sum(np.abs(psi_L) ** 2))

        # Right projection operator
        psi_R = np.zeros_like(psi)
        psi_R[split_index:] = psi[split_index:]
        prob_R = float(np.sum(np.abs(psi_R) ** 2))

        # Normalize collapsed wavefunctions
        if prob_L > 1e-12:
            psi_L = psi_L / np.sqrt(prob_L)
        if prob_R > 1e-12:
            psi_R = psi_R / np.sqrt(prob_R)

        branch_id_L = f"{parent_state.branch_id}_L_{parent_state.step}"
        branch_id_R = f"{parent_state.branch_id}_R_{parent_state.step}"

        node_L = BranchNode(
            branch_id=branch_id_L,
            parent_branch_id=parent_state.branch_id,
            step_created=parent_state.step,
            measurement_outcome="LEFT_REGION",
            branch_probability=prob_L,
            state_vector=QuantumLatticeState.from_complex_array(
                psi=psi_L, step=parent_state.step, time=parent_state.time, branch_id=branch_id_L
            ),
        )

        node_R = BranchNode(
            branch_id=branch_id_R,
            parent_branch_id=parent_state.branch_id,
            step_created=parent_state.step,
            measurement_outcome="RIGHT_REGION",
            branch_probability=prob_R,
            state_vector=QuantumLatticeState.from_complex_array(
                psi=psi_R, step=parent_state.step, time=parent_state.time, branch_id=branch_id_R
            ),
        )

        self.branches[branch_id_L] = node_L
        self.branches[branch_id_R] = node_R
        return node_L, node_R
