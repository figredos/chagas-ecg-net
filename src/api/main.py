import logging

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import settings
from src.monitoring.ab_testing import ABRouter
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
    primary_model, primary_metadata = load_model_from_registry(
        registry_root=settings.registry_root,
        model_name=settings.primary_model_name,
        model_version=settings.primary_model_version,
        device=settings.device,
    )
    secondary_model, secondary_metadata = load_model_from_registry(
        registry_root=settings.registry_root,
        model_name=settings.secondary_model_name,
        model_version=settings.secondary_model_version,
        device=settings.device,
    )

    primary_predictor = ECGPredictor(primary_model)
    secondary_predictor = ECGPredictor(secondary_model)

    app.state.ab_router = ABRouter(
        primary_predictor=primary_predictor,
        secondary_predictor=secondary_predictor,
        secondary_ratio=settings.secondary_model_ratio,
    )

    app.state.primary_metadata = primary_metadata
    app.state.secondary_metadata = secondary_metadata

    yield

    del app.state.ab_router


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="src/static"), name="static")


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
    return FileResponse("src/static/index.html")
