"""Unit Tests for Phase 11: Quantum & Statistical Mechanics Multiverse (CHIMERA v3.0)"""

import pytest
import numpy as np
from packages.quantum.models import LatticeHamiltonianConfig, QuantumLatticeState
from packages.quantum.integrator import QuantumLatticeIntegrator
from packages.quantum.decoherence import BranchingDecoherenceManager
from packages.quantum.entropy import StatisticalEntropyAnalyzer


def test_quantum_wavepacket_unitarity_preservation():
    config = LatticeHamiltonianConfig(lattice_size=32, dt=0.01)
    integrator = QuantumLatticeIntegrator(config)

    state = integrator.initialize_gaussian_wavepacket(x0=0.0, sigma=0.4, k0=2.0)
    assert state.total_probability == pytest.approx(1.0, rel=1e-6)

    # Propagate 20 steps
    for _ in range(20):
        state = integrator.step(state)

    # Unitary evolution must preserve total probability exactly
    assert state.total_probability == pytest.approx(1.0, rel=1e-6)


def test_many_worlds_branching_measurement():
    config = LatticeHamiltonianConfig(lattice_size=32, dt=0.01)
    integrator = QuantumLatticeIntegrator(config)
    state = integrator.initialize_gaussian_wavepacket(x0=0.0, sigma=0.5)

    manager = BranchingDecoherenceManager()
    node_L, node_R = manager.perform_spatial_measurement(state, split_index=16)

    assert node_L.branch_probability + node_R.branch_probability == pytest.approx(1.0, rel=1e-5)
    assert node_L.state_vector.total_probability == pytest.approx(1.0, rel=1e-5)
    assert node_R.state_vector.total_probability == pytest.approx(1.0, rel=1e-5)
    assert node_L.branch_id in manager.branches
