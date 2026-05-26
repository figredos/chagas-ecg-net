from fastapi import Request

from src.inference.predictor import ECGPredictor


def get_model(request: Request) -> ECGPredictor:
    return request.app.state.predictor
