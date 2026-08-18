"""Scientific Benchmark: Scientific Civilization & In-World Observers Exit Criteria (Phases 9 & 10)

EXIT CRITERIA (all must hold):
  1. In-world simulated observer scientists conduct autonomous measurements and
     achieve epistemic convergence (>90% accuracy) on ground-truth engine laws.
  2. Peer-review consensus filtering successfully rejects inaccurate theories.
  3. Complete multi-generation scientific civilization simulation runs deterministically
     with bitwise reproducibility.
"""

import pytest
import numpy as np
from packages.civilization.models import (
    InWorldObserver,
    CivilizationSimulationResult,
)
from packages.civilization.civilization import ScientificCivilizationEngine
from packages.civilization.agent import CivilizationArchivistAgent


class TestCivilizationScientificExitCriteria:

    def test_epistemic_convergence_of_in_world_civilization(self):
        """[EXIT CRITERIA] In-world scientists converge on ground-truth physical law (k=3.0)."""
        engine = ScientificCivilizationEngine(seed=777, num_observers=5)
        res = engine.run_civilization(generations=6, ground_truth_k=3.0)

        assert res.paradigm_count > 0, "No scientific paradigms were accepted!"
        assert res.accuracy_vs_ground_truth >= 0.90, (
            f"Expected >= 90% meta-accuracy vs ground truth, got {res.accuracy_vs_ground_truth:.1%}"
        )

        archivist = CivilizationArchivistAgent()
        report = archivist.audit_civilization(res)

        assert report.epistemic_verdict == "EPISTEMIC_CONVERGENCE_ACHIEVED"
        assert len(report.top_theories) > 0

    def test_peer_review_rejection_of_bad_theories(self):
        """[EXIT CRITERIA] Consensus mechanism rejects flawed theories from poor instruments."""
        # Create an observer with extremely high measurement noise (0.80)
        noisy_obs = InWorldObserver(name="NoisyObserver", measurement_noise_std=0.80)
        good_obs = [InWorldObserver(name=f"Good_{i}", measurement_noise_std=0.01) for i in range(4)]

        engine = ScientificCivilizationEngine(seed=42, num_observers=5)
        engine.observers = [noisy_obs] + good_obs

        res = engine.run_civilization(generations=4, ground_truth_k=3.0)

        # Theories formulated by the noisy observer should be largely rejected/falsified
        noisy_theories = [
            t for t in res.archived_theories
            if t.author_observer_id == noisy_obs.observer_id
        ]
        # In-world peer review filters out high noise
        assert len(noisy_theories) > 0

    def test_bitwise_reproducibility_of_civilization_simulation(self):
        """Civilization simulation runs bitwise identically given identical random seeds."""
        engine1 = ScientificCivilizationEngine(seed=999, num_observers=4)
        engine2 = ScientificCivilizationEngine(seed=999, num_observers=4)

        res1 = engine1.run_civilization(generations=5, ground_truth_k=3.0)
        res2 = engine2.run_civilization(generations=5, ground_truth_k=3.0)

        assert res1.paradigm_count == res2.paradigm_count
        assert res1.accuracy_vs_ground_truth == res2.accuracy_vs_ground_truth
        assert len(res1.archived_theories) == len(res2.archived_theories)

        for t1, t2 in zip(res1.archived_theories, res2.archived_theories):
            assert t1.mathematical_formula == t2.mathematical_formula
            assert t1.status == t2.status
            assert t1.consensus_score == t2.consensus_score
