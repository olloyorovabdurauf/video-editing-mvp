from fastapi import APIRouter

from app.api.v1.endpoints import billing, health, reels, uploads

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(reels.router)
api_router.include_router(billing.router)
api_router.include_router(uploads.router)
