"""Comprehensive Integration Test Suite for Prediction Persistence & Per-User Health History."""
import os
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.main import app
from app.db.session import Base, get_db
from app.db.models import PredictionRecord

# Setup isolated test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_sukshmadrishti_history.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


client = TestClient(app)

SAMPLE_FEATURES = [2.0, 130.0, 70.0, 25.0, 100.0, 28.5, 0.45, 32.0]


@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    """Recreate test database schema before each integration test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test_sukshmadrishti_history.db"):
        try:
            os.remove("./test_sukshmadrishti_history.db")
        except PermissionError:
            pass




def _register_and_login(name: str, email: str, password: str = "Password123!") -> tuple[dict[str, str], str]:
    """Helper registering user and returning authorization headers and user ID."""
    reg_res = client.post(
        "/api/v1/auth/register",
        json={"name": name, "email": email, "password": password},
    )
    assert reg_res.status_code == 201
    user_id = reg_res.json()["id"]

    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    return headers, user_id


def test_authenticated_user_prediction_persistence():
    """Integration Test: Authenticated inference automatically persists prediction record."""
    headers, user_id = _register_and_login("Dr. Alice Smith", "alice.smith@sukshmadrishti.org")

    payload = {
        "features": SAMPLE_FEATURES,
        "model_id": "rf",
        "explain": True,
    }
    predict_res = client.post("/api/v1/predict", json=payload, headers=headers)
    assert predict_res.status_code == 200
    pred_data = predict_res.json()
    assert "predicted_class" in pred_data

    # Query DB to verify persistence
    db = TestingSessionLocal()
    records = db.query(PredictionRecord).filter(PredictionRecord.user_id == user_id).all()
    assert len(records) == 1
    rec = records[0]
    assert rec.disease == "diabetes"
    assert rec.model_name == pred_data["model_used"]
    assert rec.model_version == pred_data["artifact_version"]
    assert rec.prediction == pred_data["predicted_class"]
    assert rec.source == "user_inference"
    db.close()


def test_guest_prediction_no_persistence():
    """Integration Test: Unauthenticated guest prediction runs without saving DB record."""
    payload = {
        "features": SAMPLE_FEATURES,
        "model_id": "lr",
    }
    predict_res = client.post("/api/v1/predict", json=payload)
    assert predict_res.status_code == 200

    db = TestingSessionLocal()
    count = db.query(PredictionRecord).count()
    assert count == 0
    db.close()


def test_user_history_list_retrieval():
    """Integration Test: GET /api/v1/history returns paginated prediction history for authenticated user."""
    headers, user_id = _register_and_login("Dr. Bob Jones", "bob.jones@sukshmadrishti.org")

    # Run 2 predictions
    client.post("/api/v1/predict", json={"features": SAMPLE_FEATURES, "model_id": "lr"}, headers=headers)
    client.post("/api/v1/predict", json={"features": SAMPLE_FEATURES, "model_id": "rf"}, headers=headers)

    # Fetch history
    history_res = client.get("/api/v1/history", headers=headers)
    assert history_res.status_code == 200
    hdata = history_res.json()
    assert hdata["total"] == 2
    assert len(hdata["items"]) == 2
    assert hdata["page"] == 1
    assert hdata["page_size"] == 20

    item = hdata["items"][0]
    assert "id" in item
    assert item["disease"] == "diabetes"
    assert "input_data" not in item  # Summary view excludes heavy input JSON


def test_user_isolation_history_list():
    """Integration Test: User A and User B cannot see each other's prediction history."""
    headers_a, user_id_a = _register_and_login("User A", "usera@sukshmadrishti.org")
    headers_b, user_id_b = _register_and_login("User B", "userb@sukshmadrishti.org")

    # User A runs 2 predictions
    client.post("/api/v1/predict", json={"features": SAMPLE_FEATURES, "model_id": "lr"}, headers=headers_a)
    client.post("/api/v1/predict", json={"features": SAMPLE_FEATURES, "model_id": "rf"}, headers=headers_a)

    # User B runs 1 prediction
    client.post("/api/v1/predict", json={"features": SAMPLE_FEATURES, "model_id": "svm"}, headers=headers_b)

    # Verify User A history
    res_a = client.get("/api/v1/history", headers=headers_a)
    assert res_a.status_code == 200
    assert res_a.json()["total"] == 2

    # Verify User B history
    res_b = client.get("/api/v1/history", headers=headers_b)
    assert res_b.status_code == 200
    assert res_b.json()["total"] == 1


def test_user_isolation_detail_lookup():
    """Integration Test: User B receives 404 when attempting to view User A's individual prediction record."""
    headers_a, _ = _register_and_login("User A", "usera2@sukshmadrishti.org")
    headers_b, _ = _register_and_login("User B", "userb2@sukshmadrishti.org")

    # User A creates a prediction
    client.post("/api/v1/predict", json={"features": SAMPLE_FEATURES, "model_id": "lr"}, headers=headers_a)

    # Fetch User A history to get record ID
    hist_a = client.get("/api/v1/history", headers=headers_a).json()
    record_id_a = hist_a["items"][0]["id"]

    # User A fetches own detail -> 200 OK
    detail_a = client.get(f"/api/v1/history/{record_id_a}", headers=headers_a)
    assert detail_a.status_code == 200
    assert "input_data" in detail_a.json()

    # User B attempts to fetch User A's record -> 404 Not Found (no leakage)
    detail_b = client.get(f"/api/v1/history/{record_id_a}", headers=headers_b)
    assert detail_b.status_code == 404
    assert "not found" in detail_b.json()["detail"].lower()


def test_user_isolation_delete_record():
    """Integration Test: User B cannot delete User A's prediction record. User A can delete own record."""
    headers_a, _ = _register_and_login("User A", "usera3@sukshmadrishti.org")
    headers_b, _ = _register_and_login("User B", "userb3@sukshmadrishti.org")

    # User A creates a prediction
    client.post("/api/v1/predict", json={"features": SAMPLE_FEATURES, "model_id": "rf"}, headers=headers_a)
    record_id_a = client.get("/api/v1/history", headers=headers_a).json()["items"][0]["id"]

    # User B attempts to delete User A's record -> 404 Not Found
    del_b = client.delete(f"/api/v1/history/{record_id_a}", headers=headers_b)
    assert del_b.status_code == 404

    # Record still exists for User A
    assert client.get(f"/api/v1/history/{record_id_a}", headers=headers_a).status_code == 200

    # User A deletes own record -> 200 OK
    del_a = client.delete(f"/api/v1/history/{record_id_a}", headers=headers_a)
    assert del_a.status_code == 200

    # Record is now gone
    assert client.get(f"/api/v1/history/{record_id_a}", headers=headers_a).status_code == 404


def test_unauthenticated_history_request_rejected():
    """Integration Test: Unauthenticated history requests return 401 Unauthorized."""
    assert client.get("/api/v1/history").status_code == 401
    assert client.get("/api/v1/history/some-id").status_code == 401
    assert client.delete("/api/v1/history/some-id").status_code == 401


def test_pagination_and_page_size_limit():
    """Integration Test: Pagination works with deterministic ordering and maximum page_size limit (100)."""
    headers, _ = _register_and_login("Paginated User", "page.user@sukshmadrishti.org")

    for _ in range(5):
        client.post("/api/v1/predict", json={"features": SAMPLE_FEATURES, "model_id": "lr"}, headers=headers)

    # Page 1, size 2
    res_p1 = client.get("/api/v1/history?page=1&page_size=2", headers=headers)
    assert res_p1.status_code == 200
    data1 = res_p1.json()
    assert len(data1["items"]) == 2
    assert data1["total"] == 5

    # Excessive page_size > 100 rejected by FastAPI Query validation
    res_invalid = client.get("/api/v1/history?page=1&page_size=150", headers=headers)
    assert res_invalid.status_code == 422


def test_disease_and_model_filtering():
    """Integration Test: Query filtering by disease and model_name."""
    headers, _ = _register_and_login("Filter User", "filter.user@sukshmadrishti.org")

    client.post("/api/v1/predict", json={"features": SAMPLE_FEATURES, "model_id": "lr"}, headers=headers)
    client.post("/api/v1/predict", json={"features": SAMPLE_FEATURES, "model_id": "rf"}, headers=headers)

    # Filter by disease=diabetes
    res_dis = client.get("/api/v1/history?disease=diabetes", headers=headers)
    assert res_dis.status_code == 200
    assert res_dis.json()["total"] == 2

    # Filter by model_name
    res_model = client.get("/api/v1/history?model_name=Random Forest", headers=headers)
    assert res_model.status_code == 200
    assert res_model.json()["total"] == 1
    assert res_model.json()["items"][0]["model_name"] == "Random Forest"


def test_client_supplied_user_id_ignored():
    """Integration Test: Client-supplied user_id in request body is rejected by schema validation (extra='forbid')."""
    headers, real_user_id = _register_and_login("Victim User", "victim@sukshmadrishti.org")

    # Send predict request with attempted user_id override in JSON
    payload = {
        "features": SAMPLE_FEATURES,
        "model_id": "lr",
        "user_id": "forged_victim_user_id",
    }
    res = client.post("/api/v1/predict", json=payload, headers=headers)
    # Extra fields are strictly forbidden by InferenceRequest Pydantic schema
    assert res.status_code == 422

