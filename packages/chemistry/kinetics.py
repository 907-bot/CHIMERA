"""Deterministic Mass-Action Chemical Kinetics Solver for CHIMERA Phase 6.

Solves the mass-action reaction network ODE:
  dX/dt = S · v(X, k)

where:
  - S is the stoichiometric matrix (num_species, num_reactions)
  - v_j = k_f,j * prod(X_i ^ nu_react) - k_r,j * prod(X_i ^ nu_prod)
  - Clamped/constant pool species (e.g. food/feedstock A, B) have dX_pool/dt = 0
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from packages.chemistry.models import (
    ChemicalSpecies,
    Reaction,
    ReactionNetwork,
    ChemistryState,
    KineticsSimulationResult,
)


class MassActionKineticsSolver:
    """Numerical mass-action ODE integrator using 4th-order Runge-Kutta (RK4)."""

    def __init__(self, network: ReactionNetwork):
        self.network = network
        self.species_names = network.get_species_names()
        self.S = network.get_stoichiometric_matrix()
        
        # Boolean mask of constant pool species (dX/dt = 0)
        self.is_constant_mask = np.array(
            [s.is_constant_pool for s in network.species], dtype=bool
        )

    def compute_rate_vector(self, concentrations: np.ndarray) -> np.ndarray:
        """Compute the flux vector v(X, k) for all reactions.

        Args:
            concentrations: 1D array of species concentrations in order of species_names.

        Returns:
            1D array of reaction rates v_j of length num_reactions.
        """
        # Clamp negative concentrations to 0 for numerical stability
        X = np.maximum(concentrations, 0.0)
        sp_map = {name: i for i, name in enumerate(self.species_names)}
        v = np.zeros(len(self.network.reactions), dtype=np.float64)

        for j, rxn in enumerate(self.network.reactions):
            # Forward flux
            fwd_flux = rxn.k_forward
            for r_sp, coeff in rxn.reactants.items():
                if r_sp in sp_map:
                    fwd_flux *= (X[sp_map[r_sp]] ** coeff)

            # Reverse flux
            rev_flux = 0.0
            if rxn.k_reverse > 0.0:
                rev_flux = rxn.k_reverse
                for p_sp, coeff in rxn.products.items():
                    if p_sp in sp_map:
                        rev_flux *= (X[sp_map[p_sp]] ** coeff)

            v[j] = fwd_flux - rev_flux

        return v

    def compute_derivatives(self, concentrations: np.ndarray) -> np.ndarray:
        """Compute dX/dt = S · v(X, k) with constant pools clamped to 0."""
        v = self.compute_rate_vector(concentrations)
        dXdt = self.S @ v
        # Constant pools do not change concentration
        dXdt[self.is_constant_mask] = 0.0
        return dXdt

    def step_rk4(self, concentrations: np.ndarray, dt: float) -> np.ndarray:
        """Advance concentration vector by one step dt using RK4."""
        k1 = self.compute_derivatives(concentrations)
        k2 = self.compute_derivatives(concentrations + 0.5 * dt * k1)
        k3 = self.compute_derivatives(concentrations + 0.5 * dt * k2)
        k4 = self.compute_derivatives(concentrations + dt * k3)

        next_X = concentrations + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return np.maximum(next_X, 0.0)

    def simulate(
        self,
        total_time: float = 50.0,
        dt: float = 0.01,
        custom_initial: Optional[Dict[str, float]] = None,
    ) -> KineticsSimulationResult:
        """Run full deterministic simulation and return time-series trajectory.

        Args:
            total_time:     Total simulation duration.
            dt:             Time step.
            custom_initial: Optional override for initial concentrations.

        Returns:
            KineticsSimulationResult with time points and concentration histories.
        """
        # Initial concentrations
        init_X = np.array([s.initial_concentration for s in self.network.species], dtype=np.float64)
        if custom_initial:
            for sp_name, val in custom_initial.items():
                if sp_name in self.species_names:
                    init_X[self.species_names.index(sp_name)] = val

        num_steps = int(total_time / dt)
        time_points = [0.0]
        
        # History map: sp_name -> list of values
        conc_history: Dict[str, List[float]] = {
            sp_name: [float(init_X[i])] for i, sp_name in enumerate(self.species_names)
        }

        current_X = init_X.copy()
        current_time = 0.0

        for step in range(1, num_steps + 1):
            current_X = self.step_rk4(current_X, dt)
            current_time = round(step * dt, 8)
            time_points.append(current_time)

            for i, sp_name in enumerate(self.species_names):
                conc_history[sp_name].append(float(current_X[i]))

        return KineticsSimulationResult(
            network_name=self.network.name,
            time_points=time_points,
            concentrations=conc_history,
        )


# ---------------------------------------------------------------------------
# Canonical Benchmark Chemistry Reaction Networks
# ---------------------------------------------------------------------------

def create_brusselator_network(
    A_conc: float = 1.0,
    B_conc: float = 3.0,
    X_init: float = 1.0,
    Y_init: float = 1.0,
) -> ReactionNetwork:
    """The Brusselator: classic chemical oscillator displaying limit cycle dynamics.

    Reactions:
      1. A -> X              (k1 = 1.0)
      2. 2X + Y -> 3X        (k2 = 1.0)  <-- Autocatalytic reaction!
      3. B + X -> Y + D      (k3 = 1.0)
      4. X -> E              (k4 = 1.0)

    Conditions: When B > 1 + A^2, the fixed point is unstable and a stable limit cycle forms.
    """
    species = [
        ChemicalSpecies(name="A", initial_concentration=A_conc, is_constant_pool=True),
        ChemicalSpecies(name="B", initial_concentration=B_conc, is_constant_pool=True),
        ChemicalSpecies(name="D", initial_concentration=0.0, is_constant_pool=True),
        ChemicalSpecies(name="E", initial_concentration=0.0, is_constant_pool=True),
        ChemicalSpecies(name="X", initial_concentration=X_init, is_constant_pool=False),
        ChemicalSpecies(name="Y", initial_concentration=Y_init, is_constant_pool=False),
    ]

    reactions = [
        Reaction(reaction_id="r1", name="A -> X", reactants={"A": 1}, products={"X": 1}, k_forward=1.0),
        Reaction(reaction_id="r2", name="2X + Y -> 3X (Autocatalytic)", reactants={"X": 2, "Y": 1}, products={"X": 3}, k_forward=1.0),
        Reaction(reaction_id="r3", name="B + X -> Y + D", reactants={"B": 1, "X": 1}, products={"Y": 1, "D": 1}, k_forward=1.0),
        Reaction(reaction_id="r4", name="X -> E", reactants={"X": 1}, products={"E": 1}, k_forward=1.0),
    ]

    return ReactionNetwork(name="Brusselator", species=species, reactions=reactions)


def create_formose_cycle_network(
    food_A: float = 10.0,
    seed_X: float = 0.05,
) -> ReactionNetwork:
    """Formose-like simple autocatalytic reaction cascade.

    Reactions:
      1. A + X -> 2X  (Autocatalytic amplification)
      2. X -> Waste   (Decay / output flow)
    """
    species = [
        ChemicalSpecies(name="A", initial_concentration=food_A, is_constant_pool=True),
        ChemicalSpecies(name="X", initial_concentration=seed_X, is_constant_pool=False),
        ChemicalSpecies(name="Waste", initial_concentration=0.0, is_constant_pool=True),
    ]

    reactions = [
        Reaction(reaction_id="r_auto", name="A + X -> 2X", reactants={"A": 1, "X": 1}, products={"X": 2}, k_forward=0.8),
        Reaction(reaction_id="r_decay", name="X -> Waste", reactants={"X": 1}, products={"Waste": 1}, k_forward=0.2),
    ]

    return ReactionNetwork(name="Formose_Autocatalytic_Cycle", species=species, reactions=reactions)


def create_lotka_volterra_network(
    A_conc: float = 1.0,
    X_init: float = 2.0,
    Y_init: float = 1.0,
) -> ReactionNetwork:
    """Lotka-Volterra chemical oscillator.

    Reactions:
      1. A + X -> 2X
      2. X + Y -> 2Y
      3. Y -> B
    """
    species = [
        ChemicalSpecies(name="A", initial_concentration=A_conc, is_constant_pool=True),
        ChemicalSpecies(name="B", initial_concentration=0.0, is_constant_pool=True),
        ChemicalSpecies(name="X", initial_concentration=X_init, is_constant_pool=False),
        ChemicalSpecies(name="Y", initial_concentration=Y_init, is_constant_pool=False),
    ]

    reactions = [
        Reaction(reaction_id="lv1", name="A + X -> 2X", reactants={"A": 1, "X": 1}, products={"X": 2}, k_forward=1.0),
        Reaction(reaction_id="lv2", name="X + Y -> 2Y", reactants={"X": 1, "Y": 1}, products={"Y": 2}, k_forward=1.0),
        Reaction(reaction_id="lv3", name="Y -> B", reactants={"Y": 1}, products={"B": 1}, k_forward=1.0),
    ]

    return ReactionNetwork(name="Lotka_Volterra_Chemistry", species=species, reactions=reactions)


BENCHMARK_NETWORKS = {
    "brusselator": create_brusselator_network,
    "formose": create_formose_cycle_network,
    "lotka_volterra": create_lotka_volterra_network,
}
