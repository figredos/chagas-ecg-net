import torch

import numpy as np

from dataclasses import dataclass

from src.models.base import ECGClassifier


@dataclass
class PredictorOutput:
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]


class ECGPredictor:
    def __init__(self, model: ECGClassifier) -> None:
        self.model = model
        self.device = next(iter(self.model.parameters())).device

    def predict(self, signal: np.ndarray) -> PredictorOutput:
        tensor_signal = torch.from_numpy(signal.astype(np.float32))
        tensor_signal = tensor_signal.to(self.device)
        print("model input shape:", tensor_signal.shape)

        raw = self.model.predict(tensor_signal)

        class_probs = dict(zip(self.model.class_names, raw["probabilities"]))

        return PredictorOutput(
            predicted_class=raw["label"],
            confidence=raw["confidence"],
            class_probabilities=class_probs,
        )
