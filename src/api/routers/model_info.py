from fastapi import Request, APIRouter, HTTPException

from src.api.config import settings

router = APIRouter()


@router.get("/model/info")
async def model_info(request: Request):
    if not hasattr(request.app.state, "primary_metadata") or not hasattr(
        request.app.state, "secondary_metadata"
    ):
        raise HTTPException(503, detail="Model's info hasn't been loaded yet.")

    return {
        "primary": {
            **request.app.state.primary_metadata,
            "ab_group": "a",
            "traffic_ratio": 1 - settings.secondary_model_ratio,
        },
        "secondary": {
            **request.app.state.secondary_metadata,
            "ab_group": "b",
            "traffic_ratio": settings.secondary_model_ratio,
        },
    }
