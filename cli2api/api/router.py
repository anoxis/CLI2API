"""Main API router combining all endpoints."""

from fastapi import APIRouter

from cli2api.api.v1 import chat, models, responses

api_router = APIRouter()

# Include v1 routers
api_router.include_router(chat.router, prefix="/v1", tags=["chat"])
api_router.include_router(models.router, prefix="/v1", tags=["models"])
api_router.include_router(responses.router, prefix="/v1", tags=["responses"])
