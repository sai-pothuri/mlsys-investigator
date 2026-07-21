"""
Integration tests for the FastAPI inference service.
Uses httpx TestClient — no real server needed.
"""

from __future__ import annotations
import pytest
from tests.conftest import VALID_FEATURES


class TestHealth:
    def test_status_ok(self, inference_client):
        r = inference_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_n_features(self, inference_client):
        r = inference_client.get("/health")
        assert r.json()["n_features"] == 12

    def test_model_version_present(self, inference_client):
        r = inference_client.get("/health")
        assert "model_version" in r.json()


class TestModelInfo:
    def test_status_200(self, inference_client):
        r = inference_client.get("/model/info")
        assert r.status_code == 200

    def test_twelve_feature_names(self, inference_client):
        r = inference_client.get("/model/info")
        names = r.json()["feature_names"]
        assert len(names) == 12

    def test_first_feature_name(self, inference_client):
        r = inference_client.get("/model/info")
        assert r.json()["feature_names"][0] == "account_age_days"

    def test_training_stats_has_val_accuracy(self, inference_client):
        r = inference_client.get("/model/info")
        assert "val_accuracy" in r.json()["training_stats"]

    def test_feature_distributions_present(self, inference_client):
        r = inference_client.get("/model/info")
        assert len(r.json()["feature_distributions"]) == 12


class TestPredict:
    def test_valid_request_status_200(self, inference_client):
        r = inference_client.post("/predict", json={"features": VALID_FEATURES})
        assert r.status_code == 200

    def test_prediction_is_binary(self, inference_client):
        r = inference_client.post("/predict", json={"features": VALID_FEATURES})
        assert r.json()["prediction"] in (0, 1)

    def test_probability_in_unit_interval(self, inference_client):
        r = inference_client.post("/predict", json={"features": VALID_FEATURES})
        prob = r.json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_latency_ms_positive(self, inference_client):
        r = inference_client.post("/predict", json={"features": VALID_FEATURES})
        assert r.json()["latency_ms"] > 0

    def test_request_id_echoed_when_provided(self, inference_client):
        r = inference_client.post("/predict",
                                   json={"request_id": "test-id-abc", "features": VALID_FEATURES})
        assert r.json()["request_id"] == "test-id-abc"

    def test_request_id_auto_generated_when_absent(self, inference_client):
        r = inference_client.post("/predict", json={"features": VALID_FEATURES})
        assert r.json()["request_id"]  # truthy — non-empty UUID

    def test_missing_single_feature_returns_422(self, inference_client):
        incomplete = {k: v for k, v in VALID_FEATURES.items() if k != "login_failure_rate"}
        r = inference_client.post("/predict", json={"features": incomplete})
        assert r.status_code == 422

    def test_empty_features_returns_422(self, inference_client):
        r = inference_client.post("/predict", json={"features": {}})
        assert r.status_code == 422

    def test_deterministic_same_features(self, inference_client):
        r1 = inference_client.post("/predict", json={"features": VALID_FEATURES})
        r2 = inference_client.post("/predict", json={"features": VALID_FEATURES})
        assert r1.json()["prediction"] == r2.json()["prediction"]
        assert r1.json()["probability"] == pytest.approx(r2.json()["probability"])

    def test_high_login_failure_increases_churn_probability(self, inference_client):
        low_fail = {**VALID_FEATURES, "login_failure_rate": 0.01}
        high_fail = {**VALID_FEATURES, "login_failure_rate": 0.29}

        r_low = inference_client.post("/predict", json={"features": low_fail})
        r_high = inference_client.post("/predict", json={"features": high_fail})

        assert r_high.json()["probability"] > r_low.json()["probability"]

    def test_enterprise_lowers_churn_probability(self, inference_client):
        basic = {**VALID_FEATURES, "product_category": 0.0}
        enterprise = {**VALID_FEATURES, "product_category": 2.0}

        r_basic = inference_client.post("/predict", json={"features": basic})
        r_enterprise = inference_client.post("/predict", json={"features": enterprise})

        assert r_enterprise.json()["probability"] < r_basic.json()["probability"]


class TestPredictBatch:
    def test_single_request(self, inference_client):
        payload = {"requests": [{"features": VALID_FEATURES}]}
        r = inference_client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        assert len(r.json()["results"]) == 1

    def test_hundred_requests(self, inference_client):
        payload = {"requests": [{"features": VALID_FEATURES}] * 100}
        r = inference_client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 100
        assert all(res["prediction"] in (0, 1) for res in results)

    def test_total_latency_positive(self, inference_client):
        payload = {"requests": [{"features": VALID_FEATURES}] * 10}
        r = inference_client.post("/predict/batch", json=payload)
        assert r.json()["total_latency_ms"] > 0

    def test_missing_feature_in_one_request_returns_422(self, inference_client):
        good = {"features": VALID_FEATURES}
        bad = {"features": {k: v for k, v in VALID_FEATURES.items() if k != "region"}}
        payload = {"requests": [good, bad]}
        r = inference_client.post("/predict/batch", json=payload)
        assert r.status_code == 422

    def test_request_ids_echoed(self, inference_client):
        payload = {"requests": [
            {"request_id": "batch-id-1", "features": VALID_FEATURES},
            {"request_id": "batch-id-2", "features": VALID_FEATURES},
        ]}
        r = inference_client.post("/predict/batch", json=payload)
        ids = [res["request_id"] for res in r.json()["results"]]
        assert "batch-id-1" in ids
        assert "batch-id-2" in ids
