"""P3-10 — Phase 3 API Tests

Comprehensive integration tests for Phase 3 REST API endpoints:
- GET /api/v1/benchmark/list
- POST /api/v1/benchmark/run/{world_name}
- POST /api/v1/symbolic/discover/{world_name}
- GET /api/v1/symbolic/hypotheses/{world_name}
"""

import pytest
from fastapi.testclient import TestClient
from apps.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestPhase3APIEndpoints:
    """Test suite for Phase 3 symbolic discovery and benchmark endpoints."""

    def test_list_benchmarks(self, client):
        res = client.get("/api/v1/benchmark/list")
        assert res.status_code == 200
        data = res.json()
        assert "benchmarks" in data
        names = [b["name"] for b in data["benchmarks"]]
        assert "harmonic_spring" in names
        assert "damped_oscillator" in names
        assert "keplerian_approx" in names

    def test_run_benchmark_world_valid(self, client):
        res = client.post("/api/v1/benchmark/run/harmonic_spring")
        assert res.status_code == 200
        data = res.json()
        assert data["world_name"] == "harmonic_spring"
        assert "t" in data
        assert "x" in data
        assert "v" in data
        assert "a" in data
        assert len(data["t"]) > 100

    def test_run_benchmark_world_unknown_404(self, client):
        res = client.post("/api/v1/benchmark/run/unknown_world_xyz")
        assert res.status_code == 404
        assert "Unknown benchmark world" in res.json()["detail"]

    def test_symbolic_discover_valid(self, client):
        res = client.post("/api/v1/symbolic/discover/harmonic_spring")
        assert res.status_code == 200
        data = res.json()
        assert data["world_name"] == "harmonic_spring"
        assert data["elapsed_seconds"] >= 0.0
        assert data["hypotheses_count"] >= 1
        assert data["best_hypothesis"] is not None
        best = data["best_hypothesis"]
        assert best["solver"] == "SINDy-STLSQ"
        assert len(best["candidate_equation"]) > 0
        assert best["r_squared"] > 0.95

    def test_symbolic_discover_unknown_404(self, client):
        res = client.post("/api/v1/symbolic/discover/unknown_world_xyz")
        assert res.status_code == 404

    def test_list_hypotheses_valid(self, client):
        # Discover first so registry has records
        client.post("/api/v1/symbolic/discover/damped_oscillator")

        res = client.get("/api/v1/symbolic/hypotheses/damped_oscillator")
        assert res.status_code == 200
        data = res.json()
        assert data["world_name"] == "damped_oscillator"
        assert data["count"] >= 1
        assert len(data["hypotheses"]) >= 1
        assert "candidate_equation" in data["hypotheses"][0]
