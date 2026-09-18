from fastapi.testclient import TestClient

from drynaria_api.main import create_app


def test_health_returns_stage_a_metadata() -> None:
    client = TestClient(create_app())

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "service": "drynaria-online-authentication-v2",
        "stage": "A",
        "experimental_closed_set": True,
        "open_set_calibrated": False,
    }


def test_status_reports_model_availability_without_loading(client: TestClient) -> None:
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    body = response.json()
    assert body["stage"] == "A"
    assert body["experimental_closed_set"] is True
    assert body["open_set_calibrated"] is False
    assert "model_checkpoint_available" in body
