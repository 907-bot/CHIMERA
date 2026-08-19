"""CHIMERA Quantum & Statistical Mechanics Multiverse (v3.0 - Phase 11)"""

from packages.quantum.models import LatticeHamiltonianConfig, QuantumLatticeState, BranchNode
from packages.quantum.integrator import QuantumLatticeIntegrator
from packages.quantum.decoherence import BranchingDecoherenceManager
from packages.quantum.entropy import StatisticalEntropyAnalyzer

__all__ = [
    "LatticeHamiltonianConfig",
    "QuantumLatticeState",
    "BranchNode",
    "QuantumLatticeIntegrator",
    "BranchingDecoherenceManager",
    "StatisticalEntropyAnalyzer",
]
