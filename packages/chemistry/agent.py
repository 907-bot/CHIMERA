"""Chemist Agent for Reaction Stoichiometry & Pathway Analysis for CHIMERA Phase 6.

Provides formal stoichiometric audits, conservation law checks, and stationary-state
characterization.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, ConfigDict, Field
from packages.chemistry.models import ReactionNetwork, Reaction
from packages.chemistry.hypergraph import ReactionHypergraph


class StoichiometricAudit(BaseModel):
    """Formal stoichiometric audit of a single reaction."""
    model_config = ConfigDict(frozen=True)

    reaction_id: str
    reaction_name: str
    is_autocatalytic: bool
    mass_conserved: bool
    net_stoichiometry: Dict[str, int]


class ChemistAnalysisReport(BaseModel):
    """Structured report produced by the Chemist Agent."""
    model_config = ConfigDict(frozen=True)

    network_name: str
    num_species: int
    num_reactions: int
    stoichiometry_audits: List[StoichiometricAudit]
    autocatalytic_species: List[str]
    conservation_moieties: List[Dict[str, float]]
    is_potential_oscillator: bool
    summary: str


class ChemistAgent:
    """Automated AI Chemist for reaction network stoichiometry and pathway analysis."""

    def analyze_network(self, network: ReactionNetwork) -> ChemistAnalysisReport:
        """Perform full stoichiometric and pathway audit on a ReactionNetwork.

        Args:
            network: ReactionNetwork instance.

        Returns:
            ChemistAnalysisReport with formal structural proofs.
        """
        hypergraph = ReactionHypergraph(network)
        sp_map = {s.name: s for s in network.species}

        audits: List[StoichiometricAudit] = []
        is_potential_osc = False

        for rxn in network.reactions:
            # Net stoichiometry: products - reactants
            net_st: Dict[str, int] = {}
            for sp, coeff in rxn.reactants.items():
                net_st[sp] = net_st.get(sp, 0) - coeff
            for sp, coeff in rxn.products.items():
                net_st[sp] = net_st.get(sp, 0) + coeff

            # Mass conservation check (sum(net_st * mass) == 0 for non-pool species)
            net_mass = sum(
                net_st[sp] * sp_map[sp].molecular_weight
                for sp in net_st
                if sp in sp_map and not sp_map[sp].is_constant_pool
            )
            mass_conserved = abs(net_mass) < 1e-4

            if rxn.is_autocatalytic:
                is_potential_osc = True

            audits.append(
                StoichiometricAudit(
                    reaction_id=rxn.reaction_id,
                    reaction_name=rxn.name,
                    is_autocatalytic=rxn.is_autocatalytic,
                    mass_conserved=mass_conserved,
                    net_stoichiometry=net_st,
                )
            )

        auto_species = hypergraph.identify_autocatalytic_species()
        moieties = hypergraph.compute_conservation_moieties()

        summary = (
            f"Reaction Network '{network.name}': {len(network.species)} species, "
            f"{len(network.reactions)} reactions. "
            f"Autocatalytic species detected: {auto_species}. "
            f"Conservation moieties: {len(moieties)}. "
            f"Potential for sustained oscillations/autocatalysis: {is_potential_osc}."
        )

        return ChemistAnalysisReport(
            network_name=network.name,
            num_species=len(network.species),
            num_reactions=len(network.reactions),
            stoichiometry_audits=audits,
            autocatalytic_species=auto_species,
            conservation_moieties=moieties,
            is_potential_oscillator=is_potential_osc,
            summary=summary,
        )
