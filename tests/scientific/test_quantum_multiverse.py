"""Scientific Benchmark: Quantum Dispersion, Tunneling & Entropy Growth (CHIMERA v3.0 - Phase 11)

Benchmark Goals:
1. Exact preservation of wavefunction normalization (1.0) under Hamiltonian unitary evolution.
2. Verified entropy increase during free wavepacket spatial dispersion (Statistical arrow of time).
"""

import pytest
import numpy as np
from packages.quantum.models import LatticeHamiltonianConfig
from packages.quantum.integrator import QuantumLatticeIntegrator
from packages.quantum.entropy import StatisticalEntropyAnalyzer


def test_scientific_quantum_dispersion_and_entropy():
    config = LatticeHamiltonianConfig(lattice_size=64, dx=0.1, dt=0.01, potential_type="free")
    integrator = QuantumLatticeIntegrator(config)
    analyzer = StatisticalEntropyAnalyzer()

    # Initial tight Gaussian wavepacket (low entropy)
    state = integrator.initialize_gaussian_wavepacket(x0=0.0, sigma=0.2, k0=0.0)
    initial_entropy = analyzer.compute_spatial_entropy(state)

    history = [state]
    for _ in range(50):
        state = integrator.step(state)
        history.append(state)

    # Dispersed wavepacket (higher entropy)
    final_entropy = analyzer.compute_spatial_entropy(state)
    relaxation_report = analyzer.compute_thermal_relaxation_history(history)

    print(f"\n[Quantum Entropy Benchmark] Initial S: {initial_entropy:.4f} | Final S: {final_entropy:.4f} | ΔS: {relaxation_report['entropy_change']:.4f}")

    assert state.total_probability == pytest.approx(1.0, rel=1e-5)
    assert final_entropy > initial_entropy, "Spatial dispersion must increase Shannon spatial entropy"
    assert relaxation_report["entropy_change"] > 0.5
