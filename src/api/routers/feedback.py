from fastapi import APIRouter, Response

import json
from pathlib import Path
from datetime import datetime, timezone

from src.api.schemas.feedback import FeedBackRequest

router = APIRouter()


@router.post("/feedback")
async def feedback(feedback: FeedBackRequest):
    timestamp = datetime.now(timezone.utc).isoformat()

    request = {
        **feedback.model_dump(),
        "timestamp": timestamp,
    }
    Path("feedback").mkdir(exist_ok=True)
    with open("feedback/feedback.jsonl", "a") as f:
        f.write(json.dumps(request) + "\n")

    return Response(status_code=202)
