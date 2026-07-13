import copy
import pytest

from pydantic import ValidationError

from src.api.schemas.ecg import PredictionResponse
from src.api.schemas.feedback import FeedbackRequest

PR_VALID_DICT = {
    "predicted_class": "Chagas",
    "confidence": 0.91,
    "class_probabilities": {"Chagas": 0.91, "Non-Chagas": 0.09},
    "model_name": "group_lead_cnn",
    "model_version": "v2",
    "inference_time_ms": 19.0,
}

FR_VALID_DICT = {
    "predicted_class": "Chagas",
    "true_class": "Non-Chagas",
    "confidence_score": 0.51,
}


def test_valid_prediction_response():

    prediction_response = PredictionResponse(**PR_VALID_DICT)

    assert prediction_response.predicted_class == PR_VALID_DICT["predicted_class"]
    assert prediction_response.confidence == PR_VALID_DICT["confidence"]
    assert (
        prediction_response.class_probabilities == PR_VALID_DICT["class_probabilities"]
    )
    assert prediction_response.model_name == PR_VALID_DICT["model_name"]
    assert prediction_response.model_version == PR_VALID_DICT["model_version"]
    assert prediction_response.inference_time_ms == PR_VALID_DICT["inference_time_ms"]


@pytest.mark.parametrize("key", PR_VALID_DICT.keys())
def test_invalid_prediction_response(key):
    invalid_dict = copy.deepcopy(PR_VALID_DICT)
    invalid_dict.pop(key)

    with pytest.raises(ValidationError):
        PredictionResponse(**invalid_dict)


def test_valid_feedback_request():

    feedback_request = FeedbackRequest(**FR_VALID_DICT)

    assert feedback_request.predicted_class == FR_VALID_DICT["predicted_class"]
    assert feedback_request.true_class == FR_VALID_DICT["true_class"]
    assert feedback_request.confidence_score == FR_VALID_DICT["confidence_score"]


@pytest.mark.parametrize("key", FR_VALID_DICT.keys())
def test_invalid_feedback_request(key):
    invalid_dict = copy.deepcopy(FR_VALID_DICT)
    invalid_dict.pop(key)

    with pytest.raises(ValidationError):
        FeedbackRequest(**invalid_dict)
