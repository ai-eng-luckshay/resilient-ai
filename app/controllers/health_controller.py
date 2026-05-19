from fastapi import APIRouter
from app.config.metrics import get_metrics
from app.config.settings import get_settings
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(version=settings.app_version, env=settings.env)


@router.get("/metrics")
async def metrics() -> dict:
    return get_metrics().snapshot()
