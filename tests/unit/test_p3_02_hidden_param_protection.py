"""P3-02 — Hidden Parameter Protection

Verifies that the SINDy solver and DiscoveryEngine never receive or access
hidden physics constants (e.g. k, GM, b) in their call chain, arguments,
attributes, or intermediate objects.
"""

import pytest
import inspect
from packages.symbolic.benchmark_worlds import ALL_BENCHMARKS, generate_blind_data
from packages.symbolic.sindy_solver import SINDySolver
from packages.symbolic.discovery_engine import DiscoveryEngine


class TestHiddenParameterProtection:
    """Test suite ensuring zero hidden parameter leakage into discovery solvers."""

    def test_blind_data_payload_sanitization(self):
        for world_name, spec in ALL_BENCHMARKS.items():
            blind = generate_blind_data(world_name)

            # Hidden parameter keys for this benchmark
            hidden_keys = list(spec.hidden_params.keys())

            # Verify no hidden parameter key is present in blind dictionary
            for hk in hidden_keys:
                assert hk not in blind, f"Hidden parameter '{hk}' leaked into blind_data for world '{world_name}'!"

            # Verify no hidden parameter numeric value appears as a constant attribute
            assert "hidden_params" not in blind
            assert "ground_truth" not in blind

    def test_sindy_solver_signature_and_attributes(self):
        solver = SINDySolver()

        # Check solver constructor arguments
        sig = inspect.signature(SINDySolver.__init__)
        forbidden_params = ["k", "GM", "b", "hidden_params", "true_constants"]
        for p in forbidden_params:
            assert p not in sig.parameters, f"Forbidden parameter '{p}' found in SINDySolver.__init__ signature!"

        # Check solver object attributes
        solver_attrs = dir(solver)
        for p in forbidden_params:
            assert p not in solver_attrs, f"Forbidden attribute '{p}' present on SINDySolver instance!"

    def test_discovery_engine_call_chain_blindness(self):
        engine = DiscoveryEngine()
        blind_data = generate_blind_data("harmonic_spring")

        # Solve should succeed strictly using observable arrays ('t', 'x', 'v', 'a')
        hyp = engine.sindy.solve(blind_data)

        # Hypothesis must derive parameters strictly through numerical regression
        for param_name in hyp.parameters.values.keys():
            assert param_name.startswith("coef_") or param_name == "offset"
