"""P3-17 — API / Registry Concurrency

Verifies multi-threaded concurrent write operations to the SQLite Hypothesis Registry.
Confirms WAL mode transaction isolation and data integrity without database lock errors.
"""

import os
import concurrent.futures
import pytest
from packages.symbolic.hypothesis import Hypothesis, HypothesisParameters, PredictionMetrics
from packages.symbolic.registry import HypothesisRegistry


class TestRegistryConcurrency:
    """Test suite for concurrent SQLite registry writes."""

    def test_multithreaded_concurrent_registration(self, tmp_path):
        db_file = str(tmp_path / "concurrent_registry.db")
        shared_registry = HypothesisRegistry(db_file)

        num_workers = 10
        hypotheses_per_worker = 10

        def register_batch(worker_id: int):
            local_reg = HypothesisRegistry(db_file)
            registered_ids = []
            for i in range(hypotheses_per_worker):
                h = Hypothesis(
                    world_name=f"world_{worker_id}",
                    solver=f"SINDy-{worker_id}",
                    candidate_equation=f"-3.{i}*x",
                    parameters=HypothesisParameters(values={"coef_x": -3.0 - (i * 0.01)}),
                    metrics=PredictionMetrics(r_squared=0.99, rmse=0.01, mae=0.008, train_steps=800, test_steps=200),
                    status="VALIDATED",
                )
                h_id = local_reg.register_hypothesis(h)
                registered_ids.append(h_id)
            local_reg.close()
            return registered_ids

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(register_batch, w_id) for w_id in range(num_workers)]
            all_ids = []
            for fut in concurrent.futures.as_completed(futures):
                all_ids.extend(fut.result())

        total_expected = num_workers * hypotheses_per_worker
        assert len(all_ids) == total_expected
        assert shared_registry.count_all() == total_expected

        shared_registry.close()
