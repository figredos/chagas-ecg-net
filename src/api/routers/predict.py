from pathlib import Path
import time
import dataclasses

from asyncio import to_thread

from fastapi import APIRouter, UploadFile, HTTPException, Depends, File, Form

from src.api.config import settings
from src.api.dependencies import get_model
from src.inference.parsers import ECGParser
from src.inference.predictor import ECGPredictor
from src.api.schemas.ecg import PredictionResponse
from src.monitoring.prometheus import PREDICTION_CLASS_DISTRIBUTION, PREDICTION_LATENCY

router = APIRouter()


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    hea_file: UploadFile = File(...),
    predictor: ECGPredictor = Depends(get_model),
):
    contents = await file.read()

    if file.filename is None or hea_file.filename is None:
        raise HTTPException(status_code=400, detail="File malformed")

    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File too large.")

    file_name = Path(file.filename)
    hea_file_name = Path(hea_file.filename)

    file_extension = file_name.suffix.lstrip(".")
    hea_file_extension = hea_file_name.suffix.lstrip(".")

    if file_extension != "dat" or hea_file_extension != "hea":
        raise HTTPException(status_code=415, detail="Unsupported file format")

    hea = await hea_file.read()
    base_name = file_name.stem
    ecg_signal = ECGParser.from_wfdb(
        dat_bytes=contents,
        hea_bytes=hea,
        base_name=base_name,
    )

    start = time.perf_counter()
    prediction = await to_thread(predictor.predict, ecg_signal.signal)
    process_time = (time.perf_counter() - start) * 1000

    PREDICTION_LATENCY.observe(process_time)
    PREDICTION_CLASS_DISTRIBUTION.labels(
        predicted_class=prediction.predicted_class
    ).inc()

    return PredictionResponse(
        **dataclasses.asdict(prediction),
        model_name=settings.model_name,
        model_version=settings.model_version,
        inference_time_ms=process_time,
    )
