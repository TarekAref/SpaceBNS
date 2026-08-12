"""Contract tests for the public SpaceBNS API scaffold."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_reports_safe_modes() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["data_mode"] == "synthetic-only"
    assert response.json()["action_mode"] == "simulation-only"


def test_mock_telemetry_is_explicitly_synthetic() -> None:
    response = client.get("/api/v1/mock/telemetry")

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "synthetic-public-demonstration"
    assert body["count"] > 0


def test_assessment_has_no_command_authority() -> None:
    response = client.get("/api/v1/mock/assessment")

    assert response.status_code == 200
    body = response.json()
    assert body["policy_decision"] == "PERMITTED_FOR_SIMULATION_ONLY"
    assert body["command_authority"] == "NONE"
    assert body["model_claim"] == "deterministic-public-baseline-not-trained-ai"

