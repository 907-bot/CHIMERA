"""P3-07 — Hypothesis Identity / Duplicate Handling

Defines and verifies hypothesis identity equality and distinctness.
Ensures scientifically different hypotheses are preserved as separate records.
"""

import pytest
from packages.symbolic.hypothesis import Hypothesis, HypothesisParameters, PredictionMetrics
from packages.symbolic.registry import HypothesisRegistry


class TestHypothesisIdentity:
    """Test suite for Hypothesis identity and duplicate differentiation."""

    def test_distinct_hypotheses_preserved(self):
        reg = HypothesisRegistry(":memory:")

        # Two hypotheses for same world with slightly different coefficients (e.g. from different solvers or runs)
        h1 = Hypothesis(
            world_name="harmonic_spring",
            solver="SINDy-1",
            candidate_equation="-3.01*x",
            parameters=HypothesisParameters(values={"coef_x": -3.01}),
            status="VALIDATED",
        )
        h2 = Hypothesis(
            world_name="harmonic_spring",
            solver="SINDy-2",
            candidate_equation="-2.99*x",
            parameters=HypothesisParameters(values={"coef_x": -2.99}),
            status="VALIDATED",
        )

        id1 = reg.register_hypothesis(h1)
        id2 = reg.register_hypothesis(h2)

        assert id1 != id2
        assert reg.count_all() == 2

        # Both records are distinct and queryable
        retrieved = reg.get_by_world("harmonic_spring")
        assert len(retrieved) == 2
        equations = [h.candidate_equation for h in retrieved]
        assert "-3.01*x" in equations
        assert "-2.99*x" in equations

        reg.close()

    def test_duplicate_id_insert_raises_error(self):
        reg = HypothesisRegistry(":memory:")
        h1 = Hypothesis(
            world_name="w",
            solver="SINDy",
            candidate_equation="-3.0*x",
            parameters=HypothesisParameters(values={}),
        )
        reg.register_hypothesis(h1)

        # Attempting to re-insert the identical object with the same UUID
        with pytest.raises(Exception):
            reg.register_hypothesis(h1)

        reg.close()
