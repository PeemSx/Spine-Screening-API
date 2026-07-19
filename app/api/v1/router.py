"""Version 1 route composition."""

from fastapi import APIRouter

from app.api.v1.endpoints import health, model, predictions

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(model.router, prefix="/model", tags=["model"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
