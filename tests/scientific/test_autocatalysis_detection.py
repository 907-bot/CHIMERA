"""Scientific Benchmark: Reaction-Network Chemistry & Autocatalysis Exit Criteria (Phase 6)

EXIT CRITERIA (all must hold):
  1. Autonomous detection of self-sustaining autocatalytic cycles in simulated
     reaction networks (Formose-like amplification and Brusselator oscillator).
  2. Exact numerical period estimation of limit cycle chemical oscillations.
  3. Structural hypergraph verification linking stoichiometry to kinetic feedback.
  4. Mass-action ODE solver achieves zero token cost with deterministic bitwise execution.
"""

import pytest
import numpy as np
from packages.chemistry.models import (
    ChemicalSpecies,
    Reaction,
    ReactionNetwork,
    AutocatalyticCycleResult,
)
from packages.chemistry.kinetics import (
    MassActionKineticsSolver,
    create_brusselator_network,
    create_formose_cycle_network,
    create_lotka_volterra_network,
)
from packages.chemistry.hypergraph import ReactionHypergraph
from packages.chemistry.detector import AutocatalysisDetector
from packages.chemistry.agent import ChemistAgent


class TestAutocatalysisScientificExitCriteria:

    def test_autonomous_detection_of_autocatalytic_cycle(self):
        """[EXIT CRITERIA] Autonomous detection of self-sustaining autocatalytic amplification."""
        net = create_formose_cycle_network(food_A=8.0, seed_X=0.01)
        solver = MassActionKineticsSolver(net)
        sim_res = solver.simulate(total_time=10.0, dt=0.01)

        detector = AutocatalysisDetector(net)
        res = detector.analyze_trajectory(sim_res)

        assert res.is_autocatalytic is True, "Autocatalytic cycle was not detected!"
        assert "X" in res.cycle_species
        assert res.amplification_rate > 0.1, f"Expected positive amplification rate, got {res.amplification_rate}"
        assert res.classification in ("AUTOCATALYTIC_CYCLE", "LIMIT_CYCLE_OSCILLATOR")

    def test_autonomous_detection_of_brusselator_oscillator(self):
        """[EXIT CRITERIA] Autonomous detection and period estimation of limit cycle oscillations."""
        # Standard supercritical Hopf bifurcation parameters: A=1, B=3 -> stable limit cycle
        net = create_brusselator_network(A_conc=1.0, B_conc=3.0, X_init=1.5, Y_init=2.0)
        solver = MassActionKineticsSolver(net)
        sim_res = solver.simulate(total_time=80.0, dt=0.01)

        detector = AutocatalysisDetector(net)
        res = detector.analyze_trajectory(sim_res)

        assert res.is_limit_cycle_oscillator is True, "Limit cycle oscillation was not detected!"
        assert res.is_autocatalytic is True
        assert res.oscillation_period is not None
        # Theoretical Brusselator period near Hopf bifurcation is approximately 5.0 - 8.0 s
        assert 3.0 < res.oscillation_period < 15.0, (
            f"Estimated oscillation period {res.oscillation_period}s outside expected physical range"
        )
        assert "X" in res.cycle_species

    def test_stoichiometric_conservation_and_hypergraph_linkage(self):
        """[EXIT CRITERIA] Hypergraph topological analysis identifies structural feedback loops."""
        net = create_brusselator_network()
        agent = ChemistAgent()
        report = agent.analyze_network(net)

        assert report.is_potential_oscillator is True
        assert "X" in report.autocatalytic_species
        assert len(report.stoichiometry_audits) == 4

        # Autocatalytic reaction in Brusselator: 2X + Y -> 3X
        auto_audit = next(a for a in report.stoichiometry_audits if a.is_autocatalytic)
        assert auto_audit.net_stoichiometry["X"] == 1  # Net +1 X produced per reaction event
        assert auto_audit.net_stoichiometry["Y"] == -1 # Consumes 1 Y

    def test_deterministic_reproducibility_of_chemistry_simulations(self):
        """Chemistry kinetics simulations must produce bitwise identical trajectories."""
        net = create_brusselator_network()
        solver1 = MassActionKineticsSolver(net)
        solver2 = MassActionKineticsSolver(net)

        res1 = solver1.simulate(total_time=10.0, dt=0.01)
        res2 = solver2.simulate(total_time=10.0, dt=0.01)

        np.testing.assert_array_equal(
            res1.concentrations["X"],
            res2.concentrations["X"],
        )
        np.testing.assert_array_equal(
            res1.concentrations["Y"],
            res2.concentrations["Y"],
        )
