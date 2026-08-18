"""Unit Tests for CHIMERA Chemistry Package (Phase 6).

Covers:
  - ChemicalSpecies, Reaction, and ReactionNetwork models
  - Stoichiometric matrix construction
  - MassActionKineticsSolver RK4 integration
  - ReactionHypergraph cycle and autocatalytic loop detection
  - AutocatalysisDetector on Formose and Brusselator networks
  - ChemistAgent stoichiometric analysis
"""

import pytest
import numpy as np
from packages.chemistry.models import ChemicalSpecies, Reaction, ReactionNetwork
from packages.chemistry.kinetics import (
    MassActionKineticsSolver,
    create_brusselator_network,
    create_formose_cycle_network,
    create_lotka_volterra_network,
)
from packages.chemistry.hypergraph import ReactionHypergraph
from packages.chemistry.detector import AutocatalysisDetector
from packages.chemistry.agent import ChemistAgent


class TestChemistryModels:

    def test_reaction_network_stoichiometric_matrix(self):
        # A + B -> C
        species = [
            ChemicalSpecies(name="A"),
            ChemicalSpecies(name="B"),
            ChemicalSpecies(name="C"),
        ]
        reactions = [
            Reaction(reaction_id="r1", reactants={"A": 1, "B": 1}, products={"C": 1}),
        ]
        net = ReactionNetwork(species=species, reactions=reactions)
        S = net.get_stoichiometric_matrix()

        assert S.shape == (3, 1)
        assert S[0, 0] == -1.0  # A
        assert S[1, 0] == -1.0  # B
        assert S[2, 0] == 1.0   # C

    def test_autocatalytic_reaction_property(self):
        # 2X + Y -> 3X (Autocatalytic in X)
        rxn = Reaction(
            reactants={"X": 2, "Y": 1},
            products={"X": 3},
        )
        assert rxn.is_autocatalytic is True

        # Non-autocatalytic reaction
        rxn2 = Reaction(
            reactants={"A": 1, "B": 1},
            products={"C": 1},
        )
        assert rxn2.is_autocatalytic is False


class TestMassActionKineticsSolver:

    def test_simple_decay_kinetics(self):
        # X -> Decay with k=1.0: X(t) = X0 * exp(-t)
        species = [ChemicalSpecies(name="X", initial_concentration=2.0)]
        reactions = [Reaction(reactants={"X": 1}, products={}, k_forward=1.0)]
        net = ReactionNetwork(species=species, reactions=reactions)

        solver = MassActionKineticsSolver(net)
        res = solver.simulate(total_time=2.0, dt=0.01)

        x_hist = res.concentrations["X"]
        t_hist = res.time_points

        # Analytical: 2.0 * exp(-2.0) ≈ 0.27067
        np.testing.assert_almost_equal(x_hist[-1], 2.0 * np.exp(-2.0), decimal=2)

    def test_constant_pool_remains_fixed(self):
        net = create_brusselator_network(A_conc=1.5, B_conc=3.0)
        solver = MassActionKineticsSolver(net)
        res = solver.simulate(total_time=5.0, dt=0.01)

        assert res.concentrations["A"][0] == 1.5
        assert res.concentrations["A"][-1] == 1.5
        assert res.concentrations["B"][-1] == 3.0


class TestReactionHypergraph:

    def test_hypergraph_autocatalysis_identification(self):
        net = create_formose_cycle_network()
        hg = ReactionHypergraph(net)
        auto_species = hg.identify_autocatalytic_species()

        assert "X" in auto_species

    def test_brusselator_hypergraph_cycles(self):
        net = create_brusselator_network()
        hg = ReactionHypergraph(net)
        summary = hg.summary()

        assert summary["num_species"] == 6
        assert summary["num_reactions"] == 4
        assert "X" in summary["autocatalytic_species"]


class TestAutocatalysisDetector:

    def test_detector_identifies_formose_autocatalysis(self):
        net = create_formose_cycle_network(food_A=10.0, seed_X=0.1)
        solver = MassActionKineticsSolver(net)
        sim_res = solver.simulate(total_time=5.0, dt=0.01)

        detector = AutocatalysisDetector(net)
        auto_res = detector.analyze_trajectory(sim_res)

        assert auto_res.is_autocatalytic is True
        assert "X" in auto_res.cycle_species
        assert auto_res.classification in ("AUTOCATALYTIC_CYCLE", "LIMIT_CYCLE_OSCILLATOR")

    def test_detector_identifies_brusselator_limit_cycle(self):
        # Brusselator with B=3.0 > 1 + A^2 (A=1.0 => 1+1=2) forms stable limit cycle
        net = create_brusselator_network(A_conc=1.0, B_conc=3.0, X_init=1.0, Y_init=1.0)
        solver = MassActionKineticsSolver(net)
        sim_res = solver.simulate(total_time=60.0, dt=0.01)

        detector = AutocatalysisDetector(net)
        auto_res = detector.analyze_trajectory(sim_res)

        assert auto_res.is_autocatalytic is True
        assert auto_res.is_limit_cycle_oscillator is True
        assert auto_res.oscillation_period is not None
        assert auto_res.oscillation_period > 1.0


class TestChemistAgent:

    def test_chemist_analysis_report(self):
        net = create_brusselator_network()
        agent = ChemistAgent()
        report = agent.analyze_network(net)

        assert report.network_name == "Brusselator"
        assert report.num_species == 6
        assert report.num_reactions == 4
        assert report.is_potential_oscillator is True
        assert "X" in report.autocatalytic_species
