from app.api.v1.chat import router as chat_router
from fastapi import APIRouter
from app.api.v1.health import router as health_check

api_router = APIRouter()

api_router.include_router(health_check)
api_router.include_router(chat_router)