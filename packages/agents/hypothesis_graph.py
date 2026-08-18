"""NetworkX-backed Hypothesis Provenance Graph for CHIMERA Phase 4.

Maintains a DAG (Directed Acyclic Graph) linking:
  Observations → Candidate Equations → Bull/Bear Arguments
                                     → Falsification Tests → Verdict

Per AGENTS.md Rule 6 (Immutable History):
  All nodes are append-only. No node or edge is ever deleted.
  Failed experiments and REJECTED hypotheses remain as permanent nodes.

The graph exports to JSON for persistence and can be loaded into Neo4j
in Phase 5+ when the graph scales beyond local memory.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Literal

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

from packages.agents.debate_models import (
    DebateRecord,
    BullArgument,
    BearArgument,
    CounterfactualExperiment,
    ExperimentResult,
    ArbiterVerdict,
)


class HypothesisGraph:
    """Append-only provenance DAG for hypothesis lifecycle tracking.

    Node Types:
      - world       : benchmark world specification
      - hypothesis  : candidate equation from SINDy/SR
      - argument    : Bull or Bear argument
      - experiment  : Skeptic's counterfactual
      - result      : Experiment outcome
      - verdict     : Arbiter's final decision

    Edge Types:
      - derived_from   : hypothesis → world
      - supports       : bull_argument → hypothesis
      - challenges     : bear_argument → hypothesis
      - tests          : experiment → hypothesis
      - produces       : experiment → result
      - decides        : verdict → hypothesis

    Args:
        graph_path: Optional path to save/load the graph JSON.
    """

    def __init__(self, graph_path: Optional[str] = None):
        if not _NX_AVAILABLE:
            raise ImportError(
                "networkx is required for HypothesisGraph. "
                "Install with: pip install networkx"
            )

        self.graph: nx.DiGraph = nx.DiGraph()
        self.graph_path = graph_path

        if graph_path and os.path.exists(graph_path):
            self._load(graph_path)

    # ---------------------------------------------------------------------------
    # Node Registration (Append-Only)
    # ---------------------------------------------------------------------------

    def register_world(self, world_name: str, description: str = "") -> str:
        """Add a benchmark world node if not already present."""
        node_id = f"world::{world_name}"
        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                node_type="world",
                world_name=world_name,
                description=description,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        return node_id

    def register_hypothesis(
        self,
        hypothesis_id: str,
        world_name: str,
        equation: str,
        solver: str,
        r_squared: Optional[float],
        status: str,
    ) -> str:
        """Add a hypothesis node and link it to its world."""
        node_id = f"hypothesis::{hypothesis_id}"
        world_node = self.register_world(world_name)

        self.graph.add_node(
            node_id,
            node_type="hypothesis",
            hypothesis_id=hypothesis_id,
            world_name=world_name,
            equation=equation,
            solver=solver,
            r_squared=r_squared,
            status=status,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.graph.add_edge(node_id, world_node, edge_type="derived_from")
        return node_id

    def register_bull_argument(self, bull: BullArgument) -> str:
        """Add Bull argument node and link to hypothesis."""
        node_id = f"bull::{bull.hypothesis_id}"
        hyp_node = f"hypothesis::{bull.hypothesis_id}"

        self.graph.add_node(
            node_id,
            node_type="argument",
            role="Bull",
            hypothesis_id=bull.hypothesis_id,
            confidence=bull.confidence_score,
            claim=bull.strongest_claim,
            created_at=bull.created_at,
        )
        if hyp_node in self.graph:
            self.graph.add_edge(node_id, hyp_node, edge_type="supports")
        return node_id

    def register_bear_argument(self, bear: BearArgument) -> str:
        """Add Bear argument node and link to hypothesis."""
        node_id = f"bear::{bear.hypothesis_id}"
        hyp_node = f"hypothesis::{bear.hypothesis_id}"

        self.graph.add_node(
            node_id,
            node_type="argument",
            role="Bear",
            hypothesis_id=bear.hypothesis_id,
            doubt=bear.doubt_score,
            critical_flaw=bear.critical_flaw,
            created_at=bear.created_at,
        )
        if hyp_node in self.graph:
            self.graph.add_edge(node_id, hyp_node, edge_type="challenges")
        return node_id

    def register_experiment(self, experiment: CounterfactualExperiment) -> str:
        """Add Skeptic experiment node and link to hypothesis."""
        node_id = f"experiment::{experiment.experiment_id}"
        hyp_node = f"hypothesis::{experiment.hypothesis_id}"

        self.graph.add_node(
            node_id,
            node_type="experiment",
            experiment_id=experiment.experiment_id,
            hypothesis_id=experiment.hypothesis_id,
            name=experiment.experiment_name,
            description=experiment.description,
            threshold=experiment.r2_threshold_to_survive,
            created_at=experiment.created_at,
        )
        if hyp_node in self.graph:
            self.graph.add_edge(node_id, hyp_node, edge_type="tests")
        return node_id

    def register_experiment_result(self, result: ExperimentResult) -> str:
        """Add experiment result node and link to experiment."""
        node_id = f"result::{result.experiment_id}"
        exp_node = f"experiment::{result.experiment_id}"

        self.graph.add_node(
            node_id,
            node_type="result",
            experiment_id=result.experiment_id,
            r_squared=result.r_squared_on_perturbed,
            survived=result.survived,
            interpretation=result.interpretation,
            created_at=result.created_at,
        )
        if exp_node in self.graph:
            self.graph.add_edge(exp_node, node_id, edge_type="produces")
        return node_id

    def register_verdict(self, verdict: ArbiterVerdict) -> str:
        """Add Arbiter verdict node and link to hypothesis (immutable)."""
        node_id = f"verdict::{verdict.hypothesis_id}"
        hyp_node = f"hypothesis::{verdict.hypothesis_id}"

        self.graph.add_node(
            node_id,
            node_type="verdict",
            hypothesis_id=verdict.hypothesis_id,
            verdict=verdict.verdict,
            confidence=verdict.bayesian_confidence,
            reasoning=verdict.reasoning,
            reproducibility=verdict.reproducibility_score,
            created_at=verdict.created_at,
        )
        if hyp_node in self.graph:
            self.graph.add_edge(node_id, hyp_node, edge_type="decides")
        return node_id

    def record_full_debate(self, debate: DebateRecord) -> None:
        """Register all nodes from a complete DebateRecord in one call."""
        self.register_world(debate.world_name)
        self.register_hypothesis(
            hypothesis_id=debate.hypothesis_id,
            world_name=debate.world_name,
            equation=debate.bull_argument.strongest_claim,
            solver="SINDy",
            r_squared=debate.experiment_result.r_squared_on_perturbed,
            status=debate.final_status,
        )
        self.register_bull_argument(debate.bull_argument)
        self.register_bear_argument(debate.bear_argument)
        self.register_experiment(debate.skeptic_experiment)
        self.register_experiment_result(debate.experiment_result)
        self.register_verdict(debate.arbiter_verdict)

    # ---------------------------------------------------------------------------
    # Query Methods
    # ---------------------------------------------------------------------------

    def get_hypothesis_lineage(self, hypothesis_id: str) -> Dict[str, Any]:
        """Return all nodes and edges connected to a hypothesis (full provenance).

        Args:
            hypothesis_id: UUID of the hypothesis.

        Returns:
            Dict with 'nodes' and 'edges' lists.
        """
        hyp_node = f"hypothesis::{hypothesis_id}"
        if hyp_node not in self.graph:
            return {"nodes": [], "edges": []}

        # BFS to collect all connected nodes
        connected = nx.node_connected_component(self.graph.to_undirected(), hyp_node)
        subgraph = self.graph.subgraph(connected)

        nodes = [{"id": n, **self.graph.nodes[n]} for n in subgraph.nodes]
        edges = [
            {"source": u, "target": v, **self.graph.edges[u, v]}
            for u, v in subgraph.edges
        ]
        return {"nodes": nodes, "edges": edges}

    def summary(self) -> Dict[str, int]:
        """Return counts of each node type in the graph."""
        counts: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            ntype = data.get("node_type", "unknown")
            counts[ntype] = counts.get(ntype, 0) + 1
        return counts

    def accepted_hypotheses(self) -> List[str]:
        """Return IDs of all ACCEPTED hypothesis nodes."""
        return [
            data["hypothesis_id"]
            for _, data in self.graph.nodes(data=True)
            if data.get("node_type") == "verdict" and data.get("verdict") == "ACCEPT"
        ]

    def rejected_hypotheses(self) -> List[str]:
        """Return IDs of all REJECTED hypothesis nodes (immutable — always retained)."""
        return [
            data["hypothesis_id"]
            for _, data in self.graph.nodes(data=True)
            if data.get("node_type") == "verdict" and data.get("verdict") == "REJECT"
        ]

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Export the full graph as a JSON-serialisable dict."""
        return {
            "nodes": [
                {"id": n, **{k: v for k, v in data.items() if isinstance(v, (str, int, float, bool, type(None)))}}
                for n, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **self.graph.edges[u, v]}
                for u, v in self.graph.edges
            ],
        }

    def save(self, path: Optional[str] = None) -> None:
        """Persist graph to a JSON file."""
        target = path or self.graph_path
        if not target:
            raise ValueError("No graph_path specified for save()")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def _load(self, path: str) -> None:
        """Load graph from a JSON file (append to existing graph)."""
        with open(path) as f:
            data = json.load(f)
        for node in data.get("nodes", []):
            nid = node.pop("id")
            self.graph.add_node(nid, **node)
        for edge in data.get("edges", []):
            src = edge.pop("source")
            tgt = edge.pop("target")
            self.graph.add_edge(src, tgt, **edge)
