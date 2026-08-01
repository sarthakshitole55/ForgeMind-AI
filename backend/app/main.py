from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.api.v1.documents import router as document_router
from app.config.settings import settings
from app.api.v1.chat import router as chat_router
from app.api.v1.health import router as health_router
from app.api.v1.upload import router as upload_router
from fastapi.middleware.cors import CORSMiddleware
from app.core.exceptions import ForgeMindError
from app.core.logger import logger
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ForgeMind AI Server starting up...")
    yield
    logger.info("ForgeMind AI Server shutting down...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(ForgeMindError)
async def forgemind_exception_handler(request: Request, exc: ForgeMindError):
    logger.error(f"{exc.error_type}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": {"type": exc.error_type, "message": exc.message}}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected Exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": {"type": exc.__class__.__name__, "message": "An unexpected error occurred."}}
    )

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(document_router)

@app.get("/")
def root():
    return {
        "message": "Welcome to ForgeMind AI 🚀"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }