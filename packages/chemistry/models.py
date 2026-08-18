"""Data Models for CHIMERA Reaction-Network Chemistry (Phase 6).

Defines schemas for:
  - ChemicalSpecies
  - Reaction
  - ReactionNetwork (with Stoichiometric Matrix S)
  - ChemistryState (time-series concentration snapshot)
  - AutocatalyticCycleResult (detected autocatalytic loop / oscillator)
  - KineticsSimulationResult (complete simulation trajectory)
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ChemicalSpecies(BaseModel):
    """A chemical species in the reaction network."""
    model_config = ConfigDict(frozen=True)

    name: str
    initial_concentration: float = Field(default=1.0, ge=0.0)
    molecular_weight: float = Field(default=1.0, gt=0.0)
    is_constant_pool: bool = Field(
        default=False,
        description="If True, concentration is clamped/buffered constant (e.g. constant food source A or B)"
    )


class Reaction(BaseModel):
    """A chemical reaction governed by mass-action kinetics:

    sum_i (reactants_i * S_i) -> sum_j (products_j * S_j)
    """
    model_config = ConfigDict(frozen=True)

    reaction_id: str = Field(default_factory=lambda: f"rxn_{uuid.uuid4().hex[:6]}")
    name: str = "Reaction"
    reactants: Dict[str, int] = Field(
        description="Map of species name -> stoichiometric coefficient in reactants"
    )
    products: Dict[str, int] = Field(
        description="Map of species name -> stoichiometric coefficient in products"
    )
    k_forward: float = Field(default=1.0, ge=0.0, description="Forward rate constant")
    k_reverse: float = Field(default=0.0, ge=0.0, description="Reverse rate constant (0.0 if irreversible)")
    
    @property
    def is_autocatalytic(self) -> bool:
        """True if any species appears in both reactants and products with higher product stoichiometry."""
        for sp, r_coeff in self.reactants.items():
            if sp in self.products and self.products[sp] > r_coeff:
                return True
        return False


class ReactionNetwork(BaseModel):
    """A complete reaction network with stoichiometric matrix S."""
    model_config = ConfigDict(frozen=False)

    network_id: str = Field(default_factory=lambda: f"net_{uuid.uuid4().hex[:6]}")
    name: str = "ReactionNetwork"
    species: List[ChemicalSpecies]
    reactions: List[Reaction]

    def get_species_names(self) -> List[str]:
        return [s.name for s in self.species]

    def get_stoichiometric_matrix(self) -> np.ndarray:
        """Construct the stoichiometric matrix S of shape (num_species, num_reactions).

        S_ij = product_coeff(species_i, rxn_j) - reactant_coeff(species_i, rxn_j)
        """
        sp_names = self.get_species_names()
        num_sp = len(sp_names)
        num_rxn = len(self.reactions)
        S = np.zeros((num_sp, num_rxn), dtype=np.float64)

        for j, rxn in enumerate(self.reactions):
            for sp, coeff in rxn.reactants.items():
                if sp in sp_names:
                    i = sp_names.index(sp)
                    S[i, j] -= coeff
            for sp, coeff in rxn.products.items():
                if sp in sp_names:
                    i = sp_names.index(sp)
                    S[i, j] += coeff

        return S


class ChemistryState(BaseModel):
    """State snapshot of chemical concentrations at time t."""
    model_config = ConfigDict(frozen=True)

    step: int
    time: float
    concentrations: Dict[str, float]
    reaction_rates: Optional[Dict[str, float]] = None


class AutocatalyticCycleResult(BaseModel):
    """Analysis result for an autocatalytic cycle or chemical oscillator."""
    model_config = ConfigDict(frozen=True)

    is_autocatalytic: bool
    is_limit_cycle_oscillator: bool
    cycle_species: List[str]
    amplification_rate: float = Field(description="Exponential growth rate alpha during unconstrained phase")
    oscillation_period: Optional[float] = Field(default=None, description="Period of sustained limit cycle oscillations")
    min_concentration: Dict[str, float] = Field(default_factory=dict)
    max_concentration: Dict[str, float] = Field(default_factory=dict)
    classification: Literal["AUTOCATALYTIC_CYCLE", "LIMIT_CYCLE_OSCILLATOR", "DAMPED_EQUILIBRIUM", "LINEAR_CASCADE"]
    description: str


class KineticsSimulationResult(BaseModel):
    """Complete time-series trajectory of a chemical kinetics simulation."""
    model_config = ConfigDict(frozen=True)

    network_name: str
    time_points: List[float]
    concentrations: Dict[str, List[float]]
    autocatalysis_analysis: Optional[AutocatalyticCycleResult] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
