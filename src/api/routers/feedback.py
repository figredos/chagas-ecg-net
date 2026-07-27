import logging
from asyncio import to_thread
from fastapi import APIRouter, Response

import json
from pathlib import Path
from datetime import datetime, timezone

from src.api.config import settings
from src.api.schemas.feedback import FeedbackRequest

from src.monitoring.drift import run_drift_check

router = APIRouter()


@router.post("/feedback")
async def feedback(feedback: FeedbackRequest):
    timestamp = datetime.now(timezone.utc).isoformat()

    request = {
        **feedback.model_dump(),
        "timestamp": timestamp,
    }
    Path(settings.feedback_dir).mkdir(exist_ok=True)
    with open(settings.feedback_dir / "feedback.jsonl", "a") as f:
        f.write(json.dumps(request) + "\n")

    lines = sum(1 for _ in open(settings.feedback_dir / "feedback.jsonl"))

    ab_group = feedback.ab_group or "a"

    model_name = (
        settings.primary_model_name
        if ab_group == "a"
        else settings.secondary_model_name
    )

    if lines % settings.drift_feedback_window == 0:
        result = await to_thread(
            run_drift_check,
            model_name=model_name,
            feedback_path=settings.feedback_dir,
            output_path=settings.drift_output_path,
            threshold=settings.drift_threshold,
            feedback_window=settings.drift_feedback_window,
            min_samples=settings.drift_min_samples,
        )
        if result["drifted"]:
            logging.getLogger(__name__).warning(
                "Drift detected in production predictions."
            )

    return Response(status_code=202)
