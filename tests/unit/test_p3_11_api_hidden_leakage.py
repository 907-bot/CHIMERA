"""P3-11 — API Hidden-Data Leakage

Scans OpenAPI schema, benchmark execution endpoints, and discovery endpoints
to verify zero accidental exposure of hidden physical constants (k, GM, b).
"""

import json
import pytest
from fastapi.testclient import TestClient
from apps.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestAPIHiddenDataLeakage:
    """Test suite ensuring API responses never expose hidden physics parameters."""

    def test_openapi_schema_contains_no_hidden_parameters(self, client):
        res = client.get("/openapi.json")
        assert res.status_code == 200
        schema_text = json.dumps(res.json())

        # Check for forbidden benchmark hidden parameter descriptors
        forbidden_strings = ["hidden_params", "true_constants", "ground_truth_k", "GM=50.0"]
        for s in forbidden_strings:
            assert s not in schema_text, f"Forbidden string '{s}' found in OpenAPI schema!"

    def test_benchmark_run_response_no_hidden_constants(self, client):
        for world in ["harmonic_spring", "damped_oscillator", "keplerian_approx"]:
            res = client.post(f"/api/v1/benchmark/run/{world}")
            assert res.status_code == 200
            resp_dict = res.json()

            # Forbidden keys that must never appear in public benchmark runs
            forbidden_keys = ["k", "GM", "b", "hidden_params", "seed"]
            for fk in forbidden_keys:
                assert fk not in resp_dict, f"Forbidden key '{fk}' leaked in /benchmark/run/{world} response!"

    def test_error_responses_contain_no_internal_secrets(self, client):
        # Trigger 404
        res = client.post("/api/v1/benchmark/run/nonexistent_secret_world")
        assert res.status_code == 404
        err_msg = json.dumps(res.json())
        assert "traceback" not in err_msg.lower()
        assert "password" not in err_msg.lower()
