from fastapi import Request, APIRouter, HTTPException

router = APIRouter()


@router.get("/model/info")
async def model_info(request: Request):
    if not hasattr(request.app.state, "metadata"):
        raise HTTPException(503, detail="Model's info hasn't been loaded yet.")

    return request.app.state.metadata
