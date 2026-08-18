"""Biologist Agent for Evolutionary Lineage & Adaptation Analysis for CHIMERA Phase 7.

Analyzes phylogenetic trees, selective adaptation trends, speciation rates,
and convergent evolution across parallel world runs.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, ConfigDict, Field
from packages.alife.models import ALifeSimulationResult, PhylogeneticNode, EvolutionarySnapshot


class AdaptationTrend(BaseModel):
    """Directional selective trend for a genetic trait."""
    model_config = ConfigDict(frozen=True)

    trait_name: str
    initial_mean: float
    final_mean: float
    percentage_change: float
    direction: str  # "INCREASED", "DECREASED", "STABLE"


class BiologistReport(BaseModel):
    """Structured report produced by the Biologist Agent."""
    model_config = ConfigDict(frozen=True)

    simulation_id: str
    total_generations_observed: int
    total_speciation_events: int
    adaptation_trends: List[AdaptationTrend]
    max_shannon_diversity: float
    extinction_occurred: bool
    summary: str


class BiologistAgent:
    """Automated AI Biologist for phylogenetic tree and evolutionary adaptation analysis."""

    def analyze_simulation(self, result: ALifeSimulationResult) -> BiologistReport:
        """Perform evolutionary analysis on an ALifeSimulationResult."""
        snaps = result.snapshots
        if not snaps:
            raise ValueError("Cannot analyze empty simulation result.")

        first_snap = snaps[0]
        last_snap = snaps[-1]

        # Calculate adaptation trends
        def calc_trend(name: str, v_start: float, v_end: float) -> AdaptationTrend:
            denom = abs(v_start) if abs(v_start) > 1e-4 else 1.0
            pct = ((v_end - v_start) / denom) * 100.0
            if pct > 5.0:
                direction = "INCREASED"
            elif pct < -5.0:
                direction = "DECREASED"
            else:
                direction = "STABLE"
            return AdaptationTrend(
                trait_name=name,
                initial_mean=round(v_start, 4),
                final_mean=round(v_end, 4),
                percentage_change=round(pct, 2),
                direction=direction,
            )

        trends = [
            calc_trend("Speed", first_snap.mean_speed, last_snap.mean_speed),
            calc_trend("Perception Radius", first_snap.mean_perception, last_snap.mean_perception),
            calc_trend("Energy", first_snap.mean_energy, last_snap.mean_energy),
        ]

        total_speciation = len(result.phylogenetic_tree_nodes) - 1
        max_div = max(s.shannon_diversity for s in snaps)
        extinction = last_snap.population_size == 0
        max_gen = int(math.ceil(max(s.mean_generation for s in snaps)))
        if result.total_births > 0:
            max_gen = max(1, max_gen)

        summary = (
            f"Evolutionary Run ({result.total_steps} steps): "
            f"{result.total_births} births, {result.total_deaths} deaths. "
            f"Speciation events: {total_speciation}. "
            f"Max genetic diversity: {max_div:.3f}. "
            f"Speed change: {trends[0].percentage_change:+.1f}%. "
            f"Perception change: {trends[1].percentage_change:+.1f}%."
        )

        return BiologistReport(
            simulation_id=result.simulation_id,
            total_generations_observed=max_gen,
            total_speciation_events=max(0, total_speciation),
            adaptation_trends=trends,
            max_shannon_diversity=round(max_div, 4),
            extinction_occurred=extinction,
            summary=summary,
        )
