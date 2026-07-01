from typing import Any

import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class ECGClassifier(nn.Module, ABC):
    def __init__(self, class_names: list) -> None:
        super().__init__()
        self.class_names = class_names

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _format_data(self, x: torch.Tensor) -> torch.Tensor: ...

    def predict(self, x: torch.Tensor) -> dict[str, Any]:
        self.eval()

        with torch.inference_mode():
            x = self._format_data(x)
            logits = self.forward(x)
            probs = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_index = torch.max(probs, dim=1)

            predicted_index = predicted_index.item()
            confidence_score = confidence.item()

            if not isinstance(predicted_index, int):
                raise TypeError(f"Expected int, got {type(predicted_index)}")

            label = self.class_names[predicted_index]
        return {
            "label": label,
            "confidence": confidence_score,
            "probabilities": probs.squeeze().tolist(),
            "logits": logits.squeeze().tolist(),
        }
