import time
from fastapi import Request, APIRouter, HTTPException

router = APIRouter()


@router.get("/metrics")
async def metrics(request: Request):
    if not hasattr(request.app.state, "metrics"):
        raise HTTPException(503, detail="No metrics have been initialized.")

    request_count = request.app.state.metrics["request_count"]
    avg_pred_latency = (
        (request.app.state.metrics["total_latency_ms"] / request)
        if request_count > 0
        else 0.0
    )

    total_uptime = time.time() - request.app.state.start_time

    return {
        **request.app.state.metrics,
        "avg_pred_latency_ms": avg_pred_latency,
        "total_uptime_seconds": total_uptime,
    }
