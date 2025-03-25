from fastapi import APIRouter

router = APIRouter(tags=["System"])

@router.get("/health")
async def health_check():
    return {"status": "healthy"}