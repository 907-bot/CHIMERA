"""Hypergraph & Stoichiometric Topology for CHIMERA Chemical Reaction Networks.

Represents reaction networks as directed bipartite graphs (Species <-> Reactions)
to identify:
  1. Autocatalytic cycles (directed loops where species amplify their own formation)
  2. Conservation moieties (left nullspace of S: vectors v such that v^T S = 0)
"""

from __future__ import annotations
import numpy as np
import networkx as nx
from typing import List, Dict, Set, Tuple, Any
from packages.chemistry.models import ReactionNetwork, Reaction


class ReactionHypergraph:
    """Bipartite directed hypergraph representation of chemical reaction networks."""

    def __init__(self, network: ReactionNetwork):
        self.network = network
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        """Construct the bipartite graph: Species nodes and Reaction nodes."""
        # 1. Add Species nodes
        for sp in self.network.species:
            self.graph.add_node(
                f"sp::{sp.name}",
                node_type="species",
                name=sp.name,
                is_constant=sp.is_constant_pool,
            )

        # 2. Add Reaction nodes and edges
        for rxn in self.network.reactions:
            rxn_node = f"rxn::{rxn.reaction_id}"
            self.graph.add_node(
                rxn_node,
                node_type="reaction",
                name=rxn.name,
                k_forward=rxn.k_forward,
            )

            # Reactant edges: species -> reaction
            for r_sp, coeff in rxn.reactants.items():
                self.graph.add_edge(f"sp::{r_sp}", rxn_node, stoichiometry=coeff, direction="reactant")

            # Product edges: reaction -> species
            for p_sp, coeff in rxn.products.items():
                self.graph.add_edge(rxn_node, f"sp::{p_sp}", stoichiometry=coeff, direction="product")

    def find_all_cycles(self) -> List[List[str]]:
        """Find all elementary directed cycles in the bipartite reaction graph."""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            # Filter cycles to show species names in the loop
            formatted_cycles = []
            for cyc in cycles:
                species_in_cycle = [
                    self.graph.nodes[n]["name"]
                    for n in cyc
                    if self.graph.nodes[n]["node_type"] == "species"
                ]
                if len(species_in_cycle) >= 1 and species_in_cycle not in formatted_cycles:
                    formatted_cycles.append(species_in_cycle)
            return formatted_cycles
        except Exception:
            return []

    def identify_autocatalytic_species(self) -> List[str]:
        """Identify species that appear with net positive stoichiometry in direct or cyclic loops."""
        autocatalytic_species: Set[str] = set()

        # 1. Direct autocatalysis within single reactions (e.g. 2X + Y -> 3X or A + X -> 2X)
        for rxn in self.network.reactions:
            for sp, r_coeff in rxn.reactants.items():
                if sp in rxn.products and rxn.products[sp] > r_coeff:
                    autocatalytic_species.add(sp)

        # 2. Cyclic autocatalysis (species in feedback loops)
        cycles = self.find_all_cycles()
        for cyc in cycles:
            for sp_name in cyc:
                # If species is non-constant, it participates in a feedback loop
                sp_obj = next((s for s in self.network.species if s.name == sp_name), None)
                if sp_obj and not sp_obj.is_constant_pool:
                    autocatalytic_species.add(sp_name)

        return sorted(list(autocatalytic_species))

    def compute_conservation_moieties(self) -> List[Dict[str, float]]:
        """Compute stoichiometric conservation laws (basis vectors v in left nullspace of S: v^T S = 0)."""
        S = self.network.get_stoichiometric_matrix()
        # Find left nullspace: S^T v = 0
        u, s_vals, vh = np.linalg.svd(S.T)
        tol = 1e-10
        null_mask = s_vals < tol
        
        # Additional zero singular values if matrix is rank-deficient
        num_zeros = S.shape[0] - len(s_vals)
        null_vectors = []
        
        if num_zeros > 0:
            null_vectors.extend(vh[len(s_vals):])

        sp_names = self.network.get_species_names()
        moieties = []

        for vec in null_vectors:
            # Clean up near-zero components
            cleaned = np.where(np.abs(vec) > 1e-4, vec, 0.0)
            if np.any(cleaned != 0.0):
                # Normalize so first non-zero is 1.0
                first_nonzero = cleaned[cleaned != 0.0][0]
                cleaned = cleaned / first_nonzero
                moiety_dict = {
                    sp_names[i]: float(cleaned[i])
                    for i in range(len(sp_names))
                    if abs(cleaned[i]) > 1e-4
                }
                if moiety_dict not in moieties:
                    moieties.append(moiety_dict)

        return moieties

    def summary(self) -> Dict[str, Any]:
        """Return summary metrics of the reaction hypergraph."""
        return {
            "num_species": len(self.network.species),
            "num_reactions": len(self.network.reactions),
            "num_cycles": len(self.find_all_cycles()),
            "autocatalytic_species": self.identify_autocatalytic_species(),
            "conservation_moieties": self.compute_conservation_moieties(),
        }
