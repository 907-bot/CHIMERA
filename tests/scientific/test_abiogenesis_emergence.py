"""Scientific Benchmark: Emergent Autocatalysis & Catalytic Closure (CHIMERA v4.0 - Phase 12)

Benchmark Goal:
Demonstrate stable permanent coexistence of a 4-species catalytic hypercycle (A -> B -> C -> D -> A)
under constant dilution flux, where all species remain above extinction thresholds and time-averaged
concentrations converge to the central symmetric state (0.25, 0.25, 0.25, 0.25), and verify algorithmic
detection of topological catalytic closure.
"""

import pytest
import numpy as np
from packages.abiogenesis.models import HypercycleState
from packages.abiogenesis.autocatalysis import AutocatalyticHypercycleSolver


def test_scientific_hypercycle_coexistence_and_closure():
    species = ["RNA_A", "RNA_B", "RNA_C", "RNA_D"]
    rate_constants = [1.0, 1.0, 1.0, 1.0]

    solver = AutocatalyticHypercycleSolver(species_names=species, rate_constants=rate_constants, dt=0.02)

    # Initial perturbed concentrations
    init_conc = (0.35, 0.30, 0.20, 0.15)
    state = HypercycleState(step=0, time=0.0, species_concentrations=init_conc, species_names=tuple(species))

    # Record trajectory history — long integration needed because 4-species
    # Eigen-Schuster hypercycles exhibit permanent limit cycle oscillations
    # around the symmetric fixed point (a well-known property for n >= 4).
    history = [np.array(state.species_concentrations)]
    for _ in range(2000):
        state = solver.step(state)
        history.append(np.array(state.species_concentrations))

    history_arr = np.array(history)  # (2001, 4)
    final_conc = history_arr[-1]

    # 1. Coexistence check: all species persist without extinction (> 0.05)
    min_observed_conc = np.min(history_arr)
    assert min_observed_conc > 0.05, f"Species went extinct, min concentration: {min_observed_conc}"

    # 2. Time-averaged concentration over multiple oscillation periods converges toward symmetric center
    mean_conc = np.mean(history_arr[200:], axis=0)
    expected_coexistence = np.array([0.25, 0.25, 0.25, 0.25])
    error = np.max(np.abs(mean_conc - expected_coexistence))

    print(f"\n[Hypercycle Benchmark] Mean Coexistence: {mean_conc} | Max Deviation: {error:.5f} | Min Conc: {min_observed_conc:.4f}")

    assert error < 0.08, f"Hypercycle mean did not converge to symmetric center: {mean_conc}"

    # 3. Verify Catalytic Closure graph detection
    catalysis_matrix = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ])
    closure_report = solver.detect_catalytic_closure(catalysis_matrix)
    assert closure_report["has_catalytic_closure"] is True
    assert closure_report["number_of_cycles"] >= 1
