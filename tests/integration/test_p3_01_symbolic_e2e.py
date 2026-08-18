"""P3-01 — End-to-End Symbolic Discovery Pipeline

Validates the full zero-token symbolic discovery workflow for every supported benchmark world:
Benchmark World -> Hidden Physics -> Blind Observation -> SINDy/STLSQ -> Candidate Equation -> Hypothesis -> Validation -> Registry -> API Retrieval
"""

import pytest
from packages.symbolic.benchmark_worlds import ALL_BENCHMARKS, generate_blind_data, score_against_hidden_truth
from packages.symbolic.discovery_engine import DiscoveryEngine
from packages.symbolic.registry import HypothesisRegistry
from packages.symbolic.hypothesis import Hypothesis


class TestPhase3EndToEndPipeline:
    """Integration test suite for the complete symbolic discovery pipeline."""

    @pytest.mark.parametrize("world_name", ["harmonic_spring", "damped_oscillator"])
    def test_end_to_end_discovery_pipeline(self, world_name):
        # 1. Initialize fresh in-memory hypothesis registry & discovery engine
        registry = HypothesisRegistry(":memory:")
        engine = DiscoveryEngine(registry=registry)

        # 2. Run blind discovery
        result = engine.run_discovery(world_name)

        # 3. Assert discovery succeeded and produced candidate & best hypothesis
        assert result.world_name == world_name
        assert len(result.hypotheses) > 0
        assert result.best_hypothesis is not None

        best_hyp = result.best_hypothesis
        assert best_hyp.metrics is not None
        assert best_hyp.metrics.r_squared > 0.95
        assert best_hyp.status in ("VALIDATED", "CANDIDATE")
        assert len(best_hyp.candidate_equation) > 0

        # 4. Verify hypothesis was persisted into SQLite registry
        stored_hyp = registry.get_by_id(best_hyp.id)
        assert stored_hyp is not None
        assert stored_hyp.id == best_hyp.id
        assert stored_hyp.candidate_equation == best_hyp.candidate_equation
        assert stored_hyp.status == best_hyp.status

        # 5. Verify query by world
        world_hyps = registry.get_by_world(world_name)
        assert len(world_hyps) >= 1
        assert any(h.id == best_hyp.id for h in world_hyps)

        # 6. Verify score against hidden ground truth (exit scoring only)
        scores = score_against_hidden_truth(world_name, best_hyp.parameters.values)
        if world_name == "harmonic_spring":
            # Discovered Hooke spring constant should match hidden k=3.0 closely
            assert abs(best_hyp.parameters.values.get("coef_x", 0.0) - (-3.0)) < 0.2

        registry.close()

    def test_run_all_benchmarks_batch(self):
        registry = HypothesisRegistry(":memory:")
        engine = DiscoveryEngine(registry=registry)
        results = engine.run_all_benchmarks()

        assert len(results) == len(ALL_BENCHMARKS)
        for res in results:
            assert res.best_hypothesis is not None
            assert res.elapsed_seconds > 0.0

        assert registry.count_all() >= len(ALL_BENCHMARKS)
        registry.close()
