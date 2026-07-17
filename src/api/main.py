import time
import logging

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import settings
from src.api.middleware import LoggingMiddleware
from src.inference.predictor import ECGPredictor
from src.models.model_registry import load_model_from_registry

from src.api.routers.health import router as health_router
from src.api.routers.metrics import router as metrics_router
from src.api.routers.predict import router as predict_router
from src.api.routers.feedback import router as feedback_router
from src.api.routers.model_info import router as model_info_router

from src.monitoring.prometheus import (
    ERROR_COUNT,
    PREDICTION_CLASS_DISTRIBUTION,
    PREDICTION_LATENCY,
    REQUEST_COUNT,
)


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

    yield

    del app.state.predictor


app = FastAPI(lifespan=lifespan)

logging.basicConfig(level=logging.INFO, force=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

app.include_router(feedback_router)
app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(model_info_router)
app.include_router(predict_router)


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")
