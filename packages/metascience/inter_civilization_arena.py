"""Inter-Civilization Debate & Cross-World Consensus Arena (CHIMERA v7.0 - Phase 15)"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
from packages.metascience.models import MetaInvariant


class InterCivilizationArena:
    """Orchestrates scientific debate and consensus between distinct artificial civilizations."""

    def __init__(self):
        self.recorded_debates: List[Dict[str, Any]] = []

    def evaluate_cross_world_invariant_consensus(
        self,
        candidate_invariant: str,
        universe_evaluations: Dict[str, float],  # {universe_id: r2_or_pvalue}
        consensus_threshold: float = 0.95,
    ) -> Tuple[bool, MetaInvariant]:
        """Evaluates whether an invariant is universally accepted across all civilization observations."""
        scores = list(universe_evaluations.values())
        mean_score = float(np.mean(scores)) if scores else 0.0
        is_consensus = mean_score >= consensus_threshold and all(s >= 0.85 for s in scores)

        invariant = MetaInvariant(
            invariant_id=f"meta_inv_{hash(candidate_invariant) % 100000}",
            name=f"Consensus Law: {candidate_invariant}",
            symbolic_form=candidate_invariant,
            participating_universes=tuple(universe_evaluations.keys()),
            confidence_score=mean_score,
            p_value=float(1.0 - mean_score),
        )

        self.recorded_debates.append({
            "candidate_invariant": candidate_invariant,
            "evaluations": universe_evaluations,
            "mean_score": mean_score,
            "consensus_reached": is_consensus,
        })

        return is_consensus, invariant
