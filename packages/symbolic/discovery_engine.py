"""Blind Discovery Engine Orchestrator for CHIMERA Phase 3.

Orchestrates the full zero-token symbolic discovery pipeline:

  Observatory (blind data) → SINDy / SR → Hypothesis → Validator → Registry

Usage:
    engine = DiscoveryEngine()
    result = engine.run_discovery("harmonic_spring")
    print(result.best_hypothesis.summary())

The engine never sees hidden physics constants — it orchestrates data flow
between the blind observatory, solver, and registry.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import time

from packages.symbolic.benchmark_worlds import generate_blind_data
from packages.symbolic.sindy_solver import SINDySolver
from packages.symbolic.hypothesis import Hypothesis, PredictionMetrics
from packages.symbolic.registry import HypothesisRegistry


@dataclass
class DiscoveryResult:
    """Container for the output of a single discovery run.

    Attributes:
        world_name:       Name of the benchmark world.
        hypotheses:       All candidate hypotheses generated.
        best_hypothesis:  The highest R² hypothesis (None if none generated).
        elapsed_seconds:  Total wall-clock time for the discovery run.
        registry_ids:     UUIDs of all registered hypotheses in the registry.
    """
    world_name: str
    hypotheses: List[Hypothesis] = field(default_factory=list)
    best_hypothesis: Optional[Hypothesis] = None
    elapsed_seconds: float = 0.0
    registry_ids: List[str] = field(default_factory=list)


class DiscoveryEngine:
    """Orchestrates the end-to-end blind discovery and validation pipeline.

    Args:
        registry:   HypothesisRegistry instance. If None, uses in-memory SQLite.
        sindy_threshold: STLSQ sparsity threshold for SINDy solver.
        sindy_train_ratio: Fraction of trajectory used for fitting.
    """

    def __init__(
        self,
        registry: Optional[HypothesisRegistry] = None,
        sindy_threshold: float = 0.05,
        sindy_train_ratio: float = 0.8,
    ):
        self.registry = registry or HypothesisRegistry(":memory:")
        self.sindy = SINDySolver(
            threshold=sindy_threshold,
            train_ratio=sindy_train_ratio,
        )

    def run_discovery(self, world_name: str) -> DiscoveryResult:
        """Execute the full blind discovery pipeline for a benchmark world.

        Steps:
          1. Generate blind observables (position, velocity, acceleration)
          2. Run SINDy sparse identification → CANDIDATE hypothesis
          3. Score hypothesis: R² ≥ 0.99 → VALIDATED, else → FALSIFIED
          4. Register hypothesis in append-only registry
          5. Return DiscoveryResult with best hypothesis

        The hidden physics constants are NEVER accessed here.
        Laws emerge purely from the data.

        Args:
            world_name: One of 'harmonic_spring', 'damped_oscillator', 'keplerian_approx'

        Returns:
            DiscoveryResult with all discovered hypotheses and metrics.
        """
        t_start = time.perf_counter()

        # Step 1: Obtain blind observables from Observatory
        # generate_blind_data() returns ONLY (t, x, v, a) — no hidden params
        blind_data = generate_blind_data(world_name)

        hypotheses: List[Hypothesis] = []
        registry_ids: List[str] = []

        # Step 2: SINDy sparse identification
        sindy_hyp = self.sindy.solve(blind_data)

        # Step 3: Score and promote/falsify
        if sindy_hyp.metrics is not None:
            sindy_hyp = sindy_hyp.validate(sindy_hyp.metrics, threshold=0.99)
        hypotheses.append(sindy_hyp)

        # Step 4: Register in append-only registry
        reg_id = self.registry.register_hypothesis(sindy_hyp)
        registry_ids.append(reg_id)
        if sindy_hyp.status != "CANDIDATE":
            self.registry.update_status(
                reg_id,
                sindy_hyp.status,
                metrics=sindy_hyp.metrics,
                falsification_evidence=sindy_hyp.falsification_evidence,
            )

        # Step 5: Pick best by R²
        best = max(
            hypotheses,
            key=lambda h: h.metrics.r_squared if h.metrics else -999.0,
        )

        elapsed = time.perf_counter() - t_start

        return DiscoveryResult(
            world_name=world_name,
            hypotheses=hypotheses,
            best_hypothesis=best,
            elapsed_seconds=elapsed,
            registry_ids=registry_ids,
        )

    def run_all_benchmarks(self) -> List[DiscoveryResult]:
        """Run discovery on all three canonical benchmark worlds.

        Returns:
            List of DiscoveryResult, one per world.
        """
        from packages.symbolic.benchmark_worlds import ALL_BENCHMARKS
        results = []
        for world_name in ALL_BENCHMARKS:
            result = self.run_discovery(world_name)
            results.append(result)
        return results
