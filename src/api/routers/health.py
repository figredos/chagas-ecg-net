from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/health/live")
async def live():
    return {
        "status": 200,
        "details": "App instance live.",
    }


@router.get("/health/ready")
async def ready(request: Request):
    if hasattr(request.app.state, "predictor"):
        return {
            "status": 200,
            "details": "Model loaded and ready for inference.",
        }
    else:
        raise HTTPException(503, detail="Model instance still loading.")
