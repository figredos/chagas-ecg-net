import copy
import json

from pathlib import Path

import pytest

from src.api.config import settings

VALID_FEEDBACK = {
    "predicted_class": "Chagas",
    "true_class": "non-Chagas",
    "confidence_score": 0.99,
}


def test_valid_feedback(client, tmp_path):
    settings.feedback_dir = tmp_path

    response = client.post(
        "/feedback",
        json=VALID_FEEDBACK,
    )

    assert response.status_code == 202

    lines = (settings.feedback_dir / "feedback.jsonl").read_text().strip().split("\n")
    latest_feedback = json.loads(lines[-1])

    assert latest_feedback["predicted_class"] == VALID_FEEDBACK["predicted_class"]
    assert latest_feedback["true_class"] == VALID_FEEDBACK["true_class"]
    assert latest_feedback["confidence_score"] == VALID_FEEDBACK["confidence_score"]


@pytest.mark.parametrize("key", VALID_FEEDBACK.keys())
def test_invalid_feedback(key, client):
    invalid_dict = copy.deepcopy(VALID_FEEDBACK)
    invalid_dict.pop(key)

    response = client.post("/feedback", json=invalid_dict)

    assert response.status_code == 422
