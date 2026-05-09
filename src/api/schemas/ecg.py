from typing import Literal

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    model_name: str
    model_version: str
    inference_time_ms: float
    source_format: Literal["hdf5", "wfdb"]
