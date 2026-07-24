import random

import numpy as np

from src.inference.predictor import ECGPredictor, PredictorOutput


class ABRouter:
    def __init__(
        self,
        primary_predictor: ECGPredictor,
        secondary_predictor: ECGPredictor,
        secondary_ratio: float = 0.1,
    ) -> None:
        self.primary_predictor = primary_predictor
        self.secondary_predictor = secondary_predictor
        self.secondary_ratio = secondary_ratio

    def route(self, signal: np.ndarray) -> tuple[str, PredictorOutput]:
        if random.random() < self.secondary_ratio:
            prediction = self.secondary_predictor.predict(signal)

            return ("b", prediction)

        prediction = self.primary_predictor.predict(signal)

        return ("a", prediction)
