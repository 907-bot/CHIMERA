"""P2-10 — API Integration

Tests FastAPI endpoints for Phase 2 Observatory against live components.
Uses fastapi.testclient.TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from apps.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestPhase2APIIntegration:
    """Test suite for Phase 2 Observatory REST API endpoints."""

    def test_health_check(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["service"] == "chimera-observatory-api"

    def test_sim_run_and_query_endpoints(self, client):
        # 1. Run simulation via POST /api/v1/sim/run
        req_payload = {
            "world_id": "api_test_w1",
            "seed": 42,
            "steps": 100,
            "num_particles": 4,
            "dt": 0.01,
            "integrator_type": "verlet",
        }
        res = client.post("/api/v1/sim/run", json=req_payload)
        assert res.status_code == 200
        run_data = res.json()
        assert run_data["world_id"] == "api_test_w1"
        assert run_data["steps_recorded"] == 101
        assert run_data["status"] == "RECORDED"
        assert len(run_data["initial_hash"]) > 0
        assert len(run_data["final_hash"]) > 0

        # 2. Query trajectory slice via GET /api/v1/observatory/trajectory/{world_id}
        res_traj = client.get("/api/v1/observatory/trajectory/api_test_w1?start_step=0&end_step=10")
        assert res_traj.status_code == 200
        traj_data = res_traj.json()
        assert traj_data["world_id"] == "api_test_w1"
        assert traj_data["count"] == 11
        assert len(traj_data["states"]) == 11
        assert len(traj_data["states"][0]["particles"]) == 4

        # 3. Query events via GET /api/v1/observatory/events/{world_id}
        res_events = client.get("/api/v1/observatory/events/api_test_w1")
        assert res_events.status_code == 200
        events_data = res_events.json()
        assert events_data["world_id"] == "api_test_w1"
        assert events_data["count"] >= 2  # Milestone events recorded

        # 4. Query derived features via GET /api/v1/observatory/features/{world_id}
        res_features = client.get("/api/v1/observatory/features/api_test_w1")
        assert res_features.status_code == 200
        features_data = res_features.json()
        assert features_data["world_id"] == "api_test_w1"
        assert features_data["total_steps"] == 101
        assert len(features_data["entropy_series"]) == 101
        assert len(features_data["msd_series"]) == 101

        # 5. Query blind observation via GET /api/v1/observatory/blind/{world_id}
        res_blind = client.get("/api/v1/observatory/blind/api_test_w1?step=5")
        assert res_blind.status_code == 200
        blind_data = res_blind.json()
        assert blind_data["world_id"] == "api_test_w1"
        assert blind_data["step"] == 5
        assert len(blind_data["particles_positions"]) == 4
        assert len(blind_data["particles_velocities"]) == 4
        # Verify hidden constants are not present
        assert "gravity_constant" not in blind_data
        assert "forces" not in blind_data

    def test_trajectory_not_found_returns_404(self, client):
        res = client.get("/api/v1/observatory/trajectory/nonexistent_world_404")
        assert res.status_code == 404
        assert "No trajectory records found" in res.json()["detail"]

    def test_features_not_found_returns_404(self, client):
        res = client.get("/api/v1/observatory/features/nonexistent_world_404")
        assert res.status_code == 404

    def test_blind_obs_not_found_returns_404(self, client):
        res = client.get("/api/v1/observatory/blind/nonexistent_world_404?step=0")
        assert res.status_code == 404

    def test_invalid_query_parameter_validation(self, client):
        # Negative start_step should return 422 Validation Error
        res = client.get("/api/v1/observatory/trajectory/api_test_w1?start_step=-5")
        assert res.status_code == 422
