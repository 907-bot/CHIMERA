"""3D Coarse-Grained Molecular Folding Engine (CHIMERA v4.0 - Phase 12)

Simulates 3D cubic lattice HP (Hydrophobic-Polar) protein folding.
Energy function:
    E = - epsilon_HH * (Number of non-bonded adjacent Hydrophobic-Hydrophobic contacts)
        - epsilon_HC * (Catalytic-Hydrophobic contacts)
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from packages.abiogenesis.models import Polymer3D


class MolecularFolding3D:
    """Simulates 3D cubic lattice folding of monomer sequences."""

    DIRECTIONS = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]

    def __init__(self, epsilon_hh: float = 1.0, epsilon_hc: float = 1.5):
        self.epsilon_hh = epsilon_hh
        self.epsilon_hc = epsilon_hc

    def compute_energy(self, sequence: str, coords: List[Tuple[int, int, int]]) -> float:
        """Compute HP contact energy: favorable negative energy for adjacent non-covalent hydrophobic residues."""
        n = len(sequence)
        energy = 0.0
        pos_set = set(coords)

        # Ensure no self-intersection
        if len(pos_set) < n:
            return 1e6  # Heavy penalty for lattice clashes

        for i in range(n):
            for j in range(i + 2, n):  # Non-bonded (at least 2 positions apart in chain)
                dx = abs(coords[i][0] - coords[j][0])
                dy = abs(coords[i][1] - coords[j][1])
                dz = abs(coords[i][2] - coords[j][2])
                dist_manhattan = dx + dy + dz

                if dist_manhattan == 1:  # Adjacent in 3D lattice
                    type_i = sequence[i]
                    type_j = sequence[j]
                    if type_i == "H" and type_j == "H":
                        energy -= self.epsilon_hh
                    elif (type_i == "C" and type_j == "H") or (type_i == "H" and type_j == "C"):
                        energy -= self.epsilon_hc
        return energy

    def fold_sequence(self, sequence: str, max_steps: int = 500, seed: int = 42) -> Polymer3D:
        """Folds a sequence using Monte Carlo simulated annealing on 3D lattice."""
        rng = np.random.default_rng(seed)
        n = len(sequence)

        # Initial linear conformation along x-axis
        coords: List[Tuple[int, int, int]] = [(i, 0, 0) for i in range(n)]
        current_energy = self.compute_energy(sequence, coords)

        temp = 2.0
        cooling_rate = 0.99

        for step in range(max_steps):
            # Pick a random internal residue to pivot or crankshaft
            idx = int(rng.integers(1, n))
            # Choose a new random relative direction
            dir_choice = self.DIRECTIONS[int(rng.integers(0, len(self.DIRECTIONS)))]
            prev_pos = coords[idx - 1]
            candidate_pos = (prev_pos[0] + dir_choice[0], prev_pos[1] + dir_choice[1], prev_pos[2] + dir_choice[2])

            # Propose candidate chain
            new_coords = list(coords)
            new_coords[idx] = candidate_pos

            # Shift subsequent chain to preserve bond lengths
            valid_chain = True
            for k in range(idx + 1, n):
                step_vec = (coords[k][0] - coords[k - 1][0], coords[k][1] - coords[k - 1][1], coords[k][2] - coords[k - 1][2])
                new_coords[k] = (new_coords[k - 1][0] + step_vec[0], new_coords[k - 1][1] + step_vec[1], new_coords[k - 1][2] + step_vec[2])

            new_energy = self.compute_energy(sequence, new_coords)
            delta_e = new_energy - current_energy

            if delta_e < 0 or (temp > 1e-4 and rng.random() < np.exp(-delta_e / temp)):
                coords = new_coords
                current_energy = new_energy

            temp *= cooling_rate

        is_catalytic = "C" in sequence and current_energy < -1.0

        return Polymer3D(
            id=f"poly_{hash(sequence) % 100000}",
            sequence=sequence,
            coordinates=tuple(coords),
            energy=current_energy,
            is_catalytic=is_catalytic,
        )
