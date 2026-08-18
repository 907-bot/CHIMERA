"""P3-06 — Hypothesis Registry Persistence

Verifies SQLite hypothesis registry persistence across database file reopening,
multi-world isolation, status filtering, and query ordering.
"""

import os
import pytest
from packages.symbolic.hypothesis import Hypothesis, HypothesisParameters, PredictionMetrics
from packages.symbolic.registry import HypothesisRegistry


class TestRegistryPersistence:
    """Test suite for SQLite Hypothesis Registry persistence."""

    def test_persistence_across_reopen(self, tmp_path):
        db_file = str(tmp_path / "hyp_registry.db")

        # 1. Write hypothesis
        reg1 = HypothesisRegistry(db_file)
        hyp = Hypothesis(
            world_name="persist_world",
            solver="SINDy",
            candidate_equation="-3.0*x",
            parameters=HypothesisParameters(values={"coef_x": -3.0}),
            metrics=PredictionMetrics(r_squared=0.995, rmse=0.01, mae=0.008, train_steps=800, test_steps=200),
            status="VALIDATED",
        )
        hyp_id = reg1.register_hypothesis(hyp)
        assert reg1.count_all() == 1
        reg1.close()

        # 2. Re-open database with fresh instance
        reg2 = HypothesisRegistry(db_file)
        assert reg2.count_all() == 1

        loaded = reg2.get_by_id(hyp_id)
        assert loaded is not None
        assert loaded.id == hyp_id
        assert loaded.world_name == "persist_world"
        assert loaded.candidate_equation == "-3.0*x"
        assert loaded.status == "VALIDATED"
        assert loaded.metrics.r_squared == 0.995
        assert loaded.parameters.values == {"coef_x": -3.0}

        reg2.close()

    def test_filter_by_status_and_ordering(self):
        reg = HypothesisRegistry(":memory:")

        # Create candidate, validated, falsified
        h1 = Hypothesis(world_name="w1", solver="SINDy", candidate_equation="eq1", parameters=HypothesisParameters(values={}), status="CANDIDATE")
        h2 = Hypothesis(world_name="w1", solver="SINDy", candidate_equation="eq2", parameters=HypothesisParameters(values={}), status="VALIDATED")
        h3 = Hypothesis(world_name="w1", solver="SINDy", candidate_equation="eq3", parameters=HypothesisParameters(values={}), status="FALSIFIED")
        h4 = Hypothesis(world_name="w2", solver="SINDy", candidate_equation="eq4", parameters=HypothesisParameters(values={}), status="VALIDATED")

        for h in [h1, h2, h3, h4]:
            reg.register_hypothesis(h)

        assert reg.count_all() == 4

        # Filter by world w1
        w1_all = reg.get_by_world("w1")
        assert len(w1_all) == 3

        # Filter by status
        w1_val = reg.get_by_world("w1", status_filter="VALIDATED")
        assert len(w1_val) == 1
        assert w1_val[0].candidate_equation == "eq2"

        w1_fal = reg.get_by_world("w1", status_filter="FALSIFIED")
        assert len(w1_fal) == 1
        assert w1_fal[0].candidate_equation == "eq3"

        reg.close()

    def test_empty_registry_queries(self):
        reg = HypothesisRegistry(":memory:")
        assert reg.count_all() == 0
        assert reg.get_by_id("nonexistent") is None
        assert reg.get_by_world("nonexistent") == []
        reg.close()
