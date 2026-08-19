"""Meta-Theoretical Knowledge Graph & Cross-World Synthesis (CHIMERA v7.0 - Phase 15)"""

from __future__ import annotations
from typing import Dict, Any, List, Set
import networkx as nx
from packages.metascience.models import MetaTheoreticDAG, MetaInvariant


class MetaTheoreticalGraphEngine:
    """Constructs and queries higher-order meta-theory graphs linking universe-specific models."""

    def __init__(self, graph_id: str = "meta_graph_001"):
        self.graph_id = graph_id
        self.nx_graph = nx.DiGraph()
        self.meta_invariants: List[MetaInvariant] = []

    def register_universe_theory(self, universe_id: str, law_name: str, equation: str, metadata: Dict[str, Any] = None):
        node_id = f"{universe_id}::{law_name}"
        self.nx_graph.add_node(
            node_id,
            universe_id=universe_id,
            law_name=law_name,
            equation=equation,
            metadata=metadata or {},
        )

    def link_theories(self, source_node: str, target_node: str, relation: str):
        self.nx_graph.add_edge(source_node, target_node, relation=relation)

    def register_meta_invariant(self, invariant: MetaInvariant):
        self.meta_invariants.append(invariant)
        self.nx_graph.add_node(
            invariant.invariant_id,
            type="META_INVARIANT",
            symbolic_form=invariant.symbolic_form,
            universes=list(invariant.participating_universes),
        )
        for u in invariant.participating_universes:
            for node, data in self.nx_graph.nodes(data=True):
                if data.get("universe_id") == u:
                    self.nx_graph.add_edge(invariant.invariant_id, node, relation="UNIFIES")

    def export_dag(self) -> MetaTheoreticDAG:
        nodes_dict = {n: dict(data) for n, data in self.nx_graph.nodes(data=True)}
        edges_list = [(u, v, data.get("relation", "RELATES_TO")) for u, v, data in self.nx_graph.edges(data=True)]
        return MetaTheoreticDAG(
            graph_id=self.graph_id,
            nodes=nodes_dict,
            edges=edges_list,
            meta_invariants=list(self.meta_invariants),
        )
