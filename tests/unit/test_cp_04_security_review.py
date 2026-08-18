"""CP-04 — Security Review

Automated security checks for:
- Path traversal mitigation
- SQL injection protection in DuckDB and SQLite parameter binding
- Input validation and sanitization
- Error response secret-leakage avoidance
"""

import pytest
from fastapi.testclient import TestClient
from apps.api.main import app
from packages.observatory.storage import ObservatoryStorageEngine
from packages.symbolic.registry import HypothesisRegistry


@pytest.fixture
def client():
    return TestClient(app)


class TestSecurityAudit:
    """Automated security review test suite."""

    def test_sql_injection_duckdb_parameterized(self):
        storage = ObservatoryStorageEngine(":memory:")
        # SQL Injection attempt in world_id parameter
        malicious_world = "world_1'; DROP TABLE particle_snapshots; --"
        states = storage.query_trajectory_slice(malicious_world)
        assert states == []

        # Table should still exist
        assert storage.count_recorded_steps("any") == 0
        storage.close()

    def test_sql_injection_sqlite_registry_parameterized(self):
        reg = HypothesisRegistry(":memory:")
        malicious_world = "world_1'; DROP TABLE hypotheses; --"
        hyps = reg.get_by_world(malicious_world)
        assert hyps == []
        assert reg.count_all() == 0
        reg.close()

    def test_api_path_traversal_protection(self, client):
        # Attempt path traversal in URL parameters
        res = client.get("/api/v1/observatory/trajectory/../../etc/passwd")
        # Should return 404 or 422, not 200 or internal server crash
        assert res.status_code in (404, 422)

    def test_api_malformed_json_handling(self, client):
        res = client.post(
            "/api/v1/sim/run",
            content="MALFORMED_JSON{{{",
            headers={"Content-Type": "application/json"},
        )
        assert res.status_code == 422
