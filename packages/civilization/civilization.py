"""Scientific Civilization Engine for CHIMERA Phases 9 & 10.

Simulates a multi-generation civilization of in-world scientists, peer-review consensus,
and cumulative theorem accumulation.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional
from packages.civilization.models import (
    InWorldObserver,
    CivilizationExperiment,
    CivilizationTheory,
    ScientificCivilizationState,
    CivilizationSimulationResult,
)
from packages.civilization.observers import InWorldObserverAgent


class ScientificCivilizationEngine:
    """Orchestrates an autonomous scientific civilization conducting nested in-world research."""

    def __init__(self, seed: int = 42, num_observers: int = 5):
        self.rng = np.random.default_rng(seed)
        self.num_observers = num_observers
        self.observers: List[InWorldObserver] = self._create_observers()
        self.archived_theories: List[CivilizationTheory] = []
        self.experiments_log: List[CivilizationExperiment] = []

    def _create_observers(self) -> List[InWorldObserver]:
        """Initialize the founding scientific academy of in-world observers."""
        observers = []
        specializations = ["PHYSICS", "CHEMISTRY", "ECOLOGY", "PHYSICS", "PHYSICS"]
        for i in range(self.num_observers):
            spec = specializations[i % len(specializations)]
            noise = float(self.rng.uniform(0.005, 0.02))  # Realistic instrument noise
            observers.append(
                InWorldObserver(
                    observer_id=f"obs_gen0_{i+1:02d}",
                    name=f"Scholar_{chr(65+i)}",
                    specialization=spec,
                    measurement_noise_std=noise,
                )
            )
        return observers

    def run_civilization(
        self,
        generations: int = 10,
        ground_truth_k: float = 3.0,
    ) -> CivilizationSimulationResult:
        """Simulate a scientific civilization over multiple generations."""
        timeline_snapshots: List[ScientificCivilizationState] = []

        for gen in range(generations):
            gen_experiments: List[CivilizationExperiment] = []
            gen_theories: List[CivilizationTheory] = []

            # 1. Observers conduct experiments & formulate theories
            for obs in self.observers:
                agent = InWorldObserverAgent(obs, self.rng)
                exp, theory = agent.conduct_harmonic_physics_experiment(true_k=ground_truth_k)
                gen_experiments.append(exp)
                gen_theories.append(theory)
                self.experiments_log.append(exp)

            # 2. Peer Review & Consensus Voting
            for theory in gen_theories:
                votes_for = 0
                for reviewer_obs in self.observers:
                    reviewer_agent = InWorldObserverAgent(reviewer_obs, self.rng)
                    if reviewer_agent.review_peer_theory(theory, true_k=ground_truth_k):
                        votes_for += 1

                consensus = votes_for / len(self.observers)
                status = "ACCEPTED_PARADIGM" if consensus >= 0.70 else "FALSIFIED_THEORY"

                updated_theory = theory.model_copy(update={
                    "consensus_score": round(consensus, 4),
                    "status": status,
                    "created_generation": gen,
                })
                self.archived_theories.append(updated_theory)

                # Update author observer count
                author = next((o for o in self.observers if o.observer_id == theory.author_observer_id), None)
                if author and status == "ACCEPTED_PARADIGM":
                    author.theories_accepted += 1

            # 3. Snapshot of scientific civilization state
            accepted_count = sum(1 for t in self.archived_theories if t.status == "ACCEPTED_PARADIGM")
            falsified_count = sum(1 for t in self.archived_theories if t.status == "FALSIFIED_THEORY")
            total_theories = len(self.archived_theories)
            consensus_idx = accepted_count / total_theories if total_theories > 0 else 0.0

            timeline_snapshots.append(
                ScientificCivilizationState(
                    generation=gen,
                    active_observers=len(self.observers),
                    total_experiments_conducted=len(self.experiments_log),
                    accepted_theories_count=accepted_count,
                    falsified_theories_count=falsified_count,
                    scientific_consensus_index=round(consensus_idx, 4),
                )
            )

        # 4. Meta-evaluation vs ground truth physics
        # Accuracy: fraction of accepted theories where measured k is within 5% of true_k
        accurate_theories = 0
        accepted_theories = [t for t in self.archived_theories if t.status == "ACCEPTED_PARADIGM"]

        for t in accepted_theories:
            try:
                k_val = float(t.mathematical_formula.split("*")[0].replace("F = -", "").strip())
                if abs(k_val - ground_truth_k) / ground_truth_k < 0.05:
                    accurate_theories += 1
            except Exception:
                pass

        meta_accuracy = accurate_theories / len(accepted_theories) if accepted_theories else 0.0

        return CivilizationSimulationResult(
            total_generations=generations,
            observers=self.observers,
            archived_theories=self.archived_theories,
            experiments_log=self.experiments_log,
            timeline_snapshots=timeline_snapshots,
            paradigm_count=len(accepted_theories),
            accuracy_vs_ground_truth=round(meta_accuracy, 4),
        )
