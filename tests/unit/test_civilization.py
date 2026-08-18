"""Unit Tests for CHIMERA Scientific Civilization (Phases 9 & 10).

Covers:
  - InWorldObserver model and noisy measurements
  - InWorldObserverAgent experiment execution and peer review
  - ScientificCivilizationEngine multi-generation simulation
  - CivilizationArchivistAgent meta-scientific audit
"""

import pytest
import numpy as np
from packages.civilization.models import (
    InWorldObserver,
    CivilizationExperiment,
    CivilizationTheory,
)
from packages.civilization.observers import InWorldObserverAgent
from packages.civilization.civilization import ScientificCivilizationEngine
from packages.civilization.agent import CivilizationArchivistAgent


class TestCivilizationUnits:

    def test_observer_experiment_formulation(self):
        obs = InWorldObserver(name="Observer_Isaac", measurement_noise_std=0.01)
        agent = InWorldObserverAgent(obs)

        exp, theory = agent.conduct_harmonic_physics_experiment(true_k=3.0, num_samples=50)

        assert isinstance(exp, CivilizationExperiment)
        assert isinstance(theory, CivilizationTheory)
        assert exp.measured_r_squared > 0.95
        assert "F = -" in theory.mathematical_formula
        assert obs.theories_formulated == 1

    def test_peer_review_replication(self):
        obs = InWorldObserver(name="Reviewer_Maxwell", measurement_noise_std=0.01)
        agent = InWorldObserverAgent(obs)

        accurate_theory = CivilizationTheory(
            author_observer_id="obs_1",
            title="Good Law",
            mathematical_formula="F = -3.002 * x",
            domain="PHYSICS",
            consensus_score=0.0,
        )
        assert agent.review_peer_theory(accurate_theory, true_k=3.0) is True

        bad_theory = CivilizationTheory(
            author_observer_id="obs_2",
            title="Bad Law",
            mathematical_formula="F = -0.500 * x",
            domain="PHYSICS",
            consensus_score=0.0,
        )
        assert agent.review_peer_theory(bad_theory, true_k=3.0) is False

    def test_civilization_engine_run(self):
        engine = ScientificCivilizationEngine(seed=42, num_observers=4)
        res = engine.run_civilization(generations=4, ground_truth_k=3.0)

        assert res.total_generations == 4
        assert len(res.observers) == 4
        assert len(res.timeline_snapshots) == 4
        assert res.paradigm_count > 0

    def test_archivist_agent_audit(self):
        engine = ScientificCivilizationEngine(seed=42, num_observers=4)
        res = engine.run_civilization(generations=3, ground_truth_k=3.0)
        archivist = CivilizationArchivistAgent()
        report = archivist.audit_civilization(res)

        assert report.civilization_id == res.civilization_id
        assert report.meta_accuracy > 0.8
        assert "EPISTEMIC_CONVERGENCE_ACHIEVED" in report.epistemic_verdict
