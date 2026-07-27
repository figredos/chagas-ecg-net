import pytest

from src.inference.predictor import ECGPredictor
from src.models.model_registry import load_model_from_registry

model_names = [
    "cnn_bert",
    "grouped_lead_cnn",
    "swin_transformer",
    "pn_encoder",
]


@pytest.mark.skip(reason="model_registry not available in CI")
@pytest.mark.parametrize("model_name", model_names)
def test_model(sample_ecg_signal, model_name):
    model, _ = load_model_from_registry(
        registry_root="model_registry",
        model_name=model_name,
        model_version="latest",
        load_state_dict=False,
        device="cpu",
    )

    predictor = ECGPredictor(model)

    predictor_output = predictor.predict(sample_ecg_signal)

    labels = ["chagas", "non-chagas"]

    assert predictor_output.predicted_class.lower() in labels

    assert (
        isinstance(predictor_output.confidence, float)
        and 0.0 <= predictor_output.confidence <= 1.0
    )

    assert isinstance(predictor_output.class_probabilities, dict)

    for key, value in predictor_output.class_probabilities.items():
        assert key.lower() in labels

        assert isinstance(value, float) and 0.0 <= value <= 1.0
