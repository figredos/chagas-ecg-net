import time
import dataclasses

from asyncio import to_thread

from fastapi import APIRouter, UploadFile, HTTPException, Depends

from src.api.config import settings
from src.api.dependencies import get_model
from src.inference.parsers import ECGParser
from src.inference.predictor import ECGPredictor
from src.api.schemas.ecg import PredictionResponse

router = APIRouter()


@router.post("/predict")
async def predict(
    file: UploadFile,
    format: str,
    hea_file: UploadFile | None,
    predictor: ECGPredictor = Depends(get_model),
):
    format = format.lstrip(".")
    contents = await file.read()

    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File too large.")

    if file.filename is None:
        raise HTTPException(status_code=400, detail="File malformed")

    if format == "hdf5":
        ecg_signal = ECGParser.from_hdf5(contents)
    elif format == "dat":
        format = "wfdb"
        hea = None
        if hea_file is not None:
            hea = await hea_file.read()

        ecg_signal = ECGParser.from_wfdb(dat_bytes=contents, hea_bytes=hea)
    else:
        raise HTTPException(status_code=415, detail="File must be either .dat or .hdf5")

    start = time.perf_counter()
    prediction = await to_thread(
        predictor.predict,
        ecg_signal.signal,
    )
    process_time = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        **dataclasses.asdict(prediction),
        model_name=settings.model_name,
        model_version=settings.model_version,
        inference_time_ms=process_time,
        source_format=format,
    )
