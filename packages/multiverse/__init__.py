"""CHIMERA Multiverse Engine & Cross-World Discovery Package (Phase 5 — v0.4 / v0.5)

Provides parallel world execution across World Families, Lyapunov chaos analysis,
and automated cross-world invariant detection.

Modules:
  models       : Data schemas for World Families, BranchPoints, Lyapunov & Invariant results
  chaos        : Lyapunov exponent calculation and trajectory divergence analysis
  invariants   : Cross-world invariant detector (distinguishes universal laws from historical accidents)
  orchestrator : MultiverseOrchestrator running World Families A, B, C, and D
"""

from packages.multiverse.models import (
    WorldFamilyType,
    WorldFamilySpec,
    WorldBranchSpec,
    LyapunovResult,
    InvariantResult,
    MultiverseBatchResult,
)
from packages.multiverse.chaos import LyapunovCalculator
from packages.multiverse.invariants import CrossWorldInvariantDetector
from packages.multiverse.orchestrator import MultiverseOrchestrator

__all__ = [
    "WorldFamilyType",
    "WorldFamilySpec",
    "WorldBranchSpec",
    "LyapunovResult",
    "InvariantResult",
    "MultiverseBatchResult",
    "LyapunovCalculator",
    "CrossWorldInvariantDetector",
    "MultiverseOrchestrator",
]
