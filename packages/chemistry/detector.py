"""Autonomous Autocatalysis & Chemical Oscillator Detector for CHIMERA Phase 6.

Analyzes concentration trajectories to identify:
  1. Exponential amplification phases (X(t) ~ X_0 e^(alpha * t))
  2. Sustained limit-cycle chemical oscillations (Brusselator / BZ dynamics)
  3. Stable fixed-point chemical equilibria
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from packages.chemistry.models import (
    ReactionNetwork,
    KineticsSimulationResult,
    AutocatalyticCycleResult,
)
from packages.chemistry.hypergraph import ReactionHypergraph


class AutocatalysisDetector:
    """Detects emergent autocatalysis and sustained chemical oscillations from kinetics data."""

    def __init__(self, network: Optional[ReactionNetwork] = None):
        self.network = network
        self.hypergraph = ReactionHypergraph(network) if network else None

    def analyze_trajectory(
        self,
        simulation: KineticsSimulationResult,
        min_oscillation_peaks: int = 3,
    ) -> AutocatalyticCycleResult:
        """Analyze a chemical kinetics trajectory for autocatalysis and oscillations.

        Args:
            simulation:             KineticsSimulationResult containing time points and concentrations.
            min_oscillation_peaks:  Minimum periodic peaks to confirm limit cycle.

        Returns:
            AutocatalyticCycleResult detailing classification, rates, and periods.
        """
        times = np.array(simulation.time_points)
        concs = simulation.concentrations

        autocatalytic_species: List[str] = []
        is_oscillator = False
        osc_period: Optional[float] = None
        max_alpha = 0.0

        min_concs: Dict[str, float] = {}
        max_concs: Dict[str, float] = {}

        # 1. Hypergraph structural check
        structural_auto = set(self.hypergraph.identify_autocatalytic_species()) if self.hypergraph else set()

        for sp_name, values in concs.items():
            arr = np.array(values)
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            min_concs[sp_name] = min_val
            max_concs[sp_name] = max_val

            # Skip constant pools with near-zero variation
            if (max_val - min_val) < 1e-4:
                continue

            # A. Test for initial exponential amplification: ln(X) vs t
            # Look at first 25% of trajectory or region before saturation
            growth_idx = min(len(arr) // 4, 100)
            if growth_idx > 5 and arr[0] > 1e-6 and arr[growth_idx] > arr[0]:
                y_log = np.log(np.maximum(arr[:growth_idx], 1e-9))
                x_time = times[:growth_idx]
                slope, _ = np.polyfit(x_time, y_log, 1)
                if slope > 0.05:
                    autocatalytic_species.append(sp_name)
                    max_alpha = max(max_alpha, float(slope))

            # B. Test for sustained limit-cycle oscillation
            # Find local maxima in the second half of the trajectory (after transients)
            half_idx = len(arr) // 2
            steady_arr = arr[half_idx:]
            steady_times = times[half_idx:]

            # Simple peak detection: arr[i-1] < arr[i] > arr[i+1]
            peaks = []
            for i in range(1, len(steady_arr) - 1):
                if steady_arr[i] > steady_arr[i - 1] and steady_arr[i] > steady_arr[i + 1]:
                    # Significant amplitude required (not numerical noise)
                    if (steady_arr[i] - np.min(steady_arr)) > 0.1:
                        peaks.append(i)

            if len(peaks) >= min_oscillation_peaks:
                peak_times = steady_times[peaks]
                periods = np.diff(peak_times)
                mean_period = float(np.mean(periods))
                period_std = float(np.std(periods))

                # Period must be regular (std < 10% of mean)
                if period_std < (mean_period * 0.15):
                    is_oscillator = True
                    osc_period = round(mean_period, 4)
                    if sp_name not in autocatalytic_species:
                        autocatalytic_species.append(sp_name)

        # Merge with structural autocatalysis species if present
        for sp in structural_auto:
            if sp not in autocatalytic_species:
                autocatalytic_species.append(sp)

        # Classification
        is_auto = len(autocatalytic_species) > 0

        if is_oscillator:
            classification = "LIMIT_CYCLE_OSCILLATOR"
            desc = (
                f"Sustained limit cycle chemical oscillation detected with period T={osc_period:.3f}s. "
                f"Oscillating autocatalytic species: {autocatalytic_species}."
            )
        elif is_auto:
            classification = "AUTOCATALYTIC_CYCLE"
            desc = (
                f"Self-sustaining autocatalytic amplification detected (alpha={max_alpha:.3f}/s). "
                f"Autocatalytic species: {autocatalytic_species}."
            )
        else:
            classification = "DAMPED_EQUILIBRIUM"
            desc = "Reaction network relaxes to stable static chemical equilibrium."

        return AutocatalyticCycleResult(
            is_autocatalytic=is_auto,
            is_limit_cycle_oscillator=is_oscillator,
            cycle_species=autocatalytic_species,
            amplification_rate=round(max_alpha, 4),
            oscillation_period=osc_period,
            min_concentration=min_concs,
            max_concentration=max_concs,
            classification=classification,
            description=desc,
        )
