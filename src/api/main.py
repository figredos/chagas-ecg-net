from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.api.config import settings
from src.inference.predictor import ECGPredictor
from src.models.model_registry import load_model_from_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, metadata = load_model_from_registry(
        registry_root=settings.registry_root,
        model_name=settings.model_name,
        model_version=settings.model_version,
        device=settings.device,
    )

    app.state.predictor = ECGPredictor(model=model)
    app.state.metadata = metadata

    app.state.metrics = {"request_count": 0, "error_count": 0, "total_latency_ms": 0.0}

    yield

    del app.state.predictor


app = FastAPI(lifespan=lifespan)
