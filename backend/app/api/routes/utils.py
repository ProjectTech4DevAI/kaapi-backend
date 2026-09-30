from fastapi import APIRouter

router = APIRouter(prefix="/utils", tags=["utils"])


@router.get("/health", include_in_schema=False)
async def health_check() -> bool:
    return True
