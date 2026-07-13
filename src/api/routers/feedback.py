from fastapi import APIRouter, Response

import json
from pathlib import Path
from datetime import datetime, timezone

from src.api.config import settings
from src.api.schemas.feedback import FeedbackRequest

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

    return Response(status_code=202)
