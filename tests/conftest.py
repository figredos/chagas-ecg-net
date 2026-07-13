import time

import pytest
import numpy as np

from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.dependencies import get_model
from src.inference.predictor import ECGPredictor, PredictorOutput


@pytest.fixture
def client() -> TestClient:
    test_client = TestClient(app)

    app.state.start_time = time.time()
    app.state.metrics = {
        "request_count": 0,
        "error_count": 0,
        "total_latency_ms": 0.0,
        "avg_pred_latency_ms": 0.0,
        "total_uptime_seconds": 0.0,
    }
    return test_client


@pytest.fixture
def mock_predictor():
    mock = MagicMock(spec=ECGPredictor)

    mock_predictor_output = PredictorOutput(
        predicted_class="Chagas",
        confidence=0.91,
        class_probabilities={"Chagas": 0.91, "Non-Chagas": 0.09},
    )

    mock.predict.return_value = mock_predictor_output

    app.dependency_overrides[get_model] = lambda: mock
    app.state.predictor = mock

    yield mock

    app.dependency_overrides.clear()


@pytest.fixture()
def sample_ecg_signal():
    return np.random.randn(12, 734)
