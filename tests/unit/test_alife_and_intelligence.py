"""Unit Tests for CHIMERA Artificial Life (Phase 7) and Embodied Intelligence (Phase 8).

Covers:
  - Genome mutation and genetic distance
  - MetabolicEngine energy burn, feeding, and reproduction
  - EvolutionaryEngine population dynamics and speciation
  - BiologistAgent report generation
  - NeuralAgentController forward inference
  - EmergenceDetector (Swarm Polarization and Transfer Entropy)
  - SocialScientistAgent analysis
"""

import pytest
import numpy as np
from packages.core.models import Vector2D, Boundary
from packages.alife.models import (
    Genome,
    Organism,
    Environment,
    FoodPatch,
    PhylogeneticNode,
)
from packages.alife.metabolism import MetabolicEngine
from packages.alife.evolution import EvolutionaryEngine
from packages.alife.agent import BiologistAgent

from packages.intelligence.models import (
    NeuralPolicy,
    SensoryObservation,
    AgentAction,
    InformationMetrics,
    SocialSimulationResult,
)
from packages.intelligence.controller import NeuralAgentController
from packages.intelligence.information import EmergenceDetector
from packages.intelligence.agent import SocialScientistAgent


class TestArtificialLifeUnits:

    def test_genome_mutation(self):
        rng = np.random.default_rng(42)
        g = Genome(speed=2.0, mutation_rate=1.0)  # Guarantee mutation
        g_mut = g.mutate(rng)

        assert g_mut.speed != g.speed
        assert g.genetic_distance(g_mut) > 0.0

    def test_metabolic_engine_consumption(self):
        rng = np.random.default_rng(42)
        engine = MetabolicEngine()

        env = Environment(
            food_patches=[FoodPatch(id=1, position=Vector2D(x=10.0, y=10.0), energy_value=20.0)]
        )
        org = Organism(
            position=Vector2D(x=10.5, y=10.5),  # Within feeding distance
            energy=10.0,
            genome=Genome(perception_radius=20.0, speed=1.0),
        )

        updated_org, offspring, food_id = engine.update_organism(org, env, dt=0.1, rng=rng)

        assert food_id == 1
        assert updated_org.energy > 10.0  # Energy gained

    def test_evolutionary_engine_run(self):
        engine = EvolutionaryEngine(seed=42)
        res = engine.run_simulation(initial_population_size=10, total_steps=30)

        assert res.total_steps == 30
        assert len(res.snapshots) == 30
        assert len(res.phylogenetic_tree_nodes) >= 1

    def test_biologist_agent_report(self):
        engine = EvolutionaryEngine(seed=101)
        res = engine.run_simulation(initial_population_size=8, total_steps=20)
        agent = BiologistAgent()
        report = agent.analyze_simulation(res)

        assert report.simulation_id == res.simulation_id
        assert len(report.adaptation_trends) == 3


class TestIntelligenceUnits:

    def test_neural_controller_forward(self):
        rng = np.random.default_rng(42)
        policy = NeuralPolicy.create_random(rng)
        obs = SensoryObservation(
            food_dx=10.0,
            food_dy=-5.0,
            nearest_agent_dx=2.0,
            nearest_agent_dy=3.0,
            current_energy=25.0,
        )

        action = NeuralAgentController.forward(policy, obs, max_speed=2.5)

        assert isinstance(action, AgentAction)
        assert -2.5 <= action.move_dx <= 2.5
        assert -2.5 <= action.move_dy <= 2.5
        assert 0.0 <= action.broadcast_signal <= 1.0

    def test_emergence_detector_polarization(self):
        detector = EmergenceDetector()

        # All aligned velocities -> Polarization = 1.0
        aligned_vels = [(1.0, 0.0), (2.0, 0.0), (0.5, 0.0)]
        phi = detector.calculate_polarization(aligned_vels)
        np.testing.assert_almost_equal(phi, 1.0, decimal=3)

        # Opposing velocities -> Polarization ≈ 0.0
        opposing_vels = [(1.0, 0.0), (-1.0, 0.0)]
        phi_opp = detector.calculate_polarization(opposing_vels)
        np.testing.assert_almost_equal(phi_opp, 0.0, decimal=3)

    def test_transfer_entropy_computation(self):
        detector = EmergenceDetector()

        # Perfectly correlated time series with lag 1
        src = [float(i % 4) for i in range(50)]
        tgt = [0.0] + src[:-1]

        te = detector.calculate_transfer_entropy(src, tgt)
        assert te > 0.0  # Positive information transfer

    def test_social_scientist_agent(self):
        detector = EmergenceDetector()
        metrics = InformationMetrics(
            transfer_entropy=0.35,
            mutual_information=0.25,
            swarm_polarization=0.85,
            is_collective_emergence=True,
            classification="COLLECTIVE_COORDINATION",
            description="Test flock",
        )
        sim_res = SocialSimulationResult(
            total_steps=50,
            num_agents=10,
            information_metrics=metrics,
            mean_energy_history=[20.0] * 50,
            polarization_history=[0.85] * 50,
        )

        agent = SocialScientistAgent()
        report = agent.analyze_social_dynamics(sim_res)

        assert report.is_collective_intelligence is True
        assert report.swarm_polarization == 0.85
