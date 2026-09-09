from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, APIRouter
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.chat import router as chat_router
from app.api.v1.documents import router as document_router
from app.api.v1.health import router as health_router
from app.api.v1.maintenance import router as maintenance_router
from app.api.v1.upload import router as upload_router
from app.config.settings import settings
from app.core.exceptions import ForgeMindError
from app.core.logger import logger
from app.core.middleware import RequestIDMiddleware
from app.core.observability import ObservabilityService


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ForgeMind AI Server starting up...")
    yield
    logger.info("ForgeMind AI Server shutting down...")
    try:
        ObservabilityService.flush()
    except Exception as e:
        logger.warning(f"Error flushing observability during shutdown: {e}")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise Multi-Agent RAG & Reasoning Engine API",
    lifespan=lifespan,
)

# 1. Request ID / Correlation Middleware (runs before all routes)
app.add_middleware(RequestIDMiddleware)

# 2. CORS Middleware with configured origins and exposed headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# -----------------------------------------------------------------------------
# Unified Error Handlers
# -----------------------------------------------------------------------------

@app.exception_handler(ForgeMindError)
async def forgemind_exception_handler(request: Request, exc: ForgeMindError):
    logger.error(f"{exc.error_type}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {"type": exc.error_type, "message": exc.message},
            "detail": exc.message,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    error_messages = []
    for err in errors:
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        error_messages.append(f"{field}: {msg}" if field else msg)
    combined_message = "; ".join(error_messages)
    logger.warning(f"Request validation failed: {combined_message}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {"type": "ValidationError", "message": combined_message},
            "detail": combined_message,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail_str = str(exc.detail) if isinstance(exc.detail, str) else str(exc.detail)
    logger.warning(f"HTTP {exc.status_code}: {detail_str}")
    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content={
            "success": False,
            "error": {"type": "HTTPException", "message": detail_str},
            "detail": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected Exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {"type": exc.__class__.__name__, "message": "An unexpected error occurred."},
            "detail": "An unexpected error occurred.",
        },
    )


# -----------------------------------------------------------------------------
# Routing: Standardized API v1 + Legacy Compatibility Paths
# -----------------------------------------------------------------------------

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health_router)
api_v1_router.include_router(chat_router)
api_v1_router.include_router(upload_router)
api_v1_router.include_router(document_router)
api_v1_router.include_router(maintenance_router)

# Mount /api/v1 as canonical public API
app.include_router(api_v1_router)

# Mount unversioned routes for backward compatibility with frontend and existing clients
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(document_router)
app.include_router(maintenance_router)


@app.get("/", summary="Root health & welcome check", tags=["General"])
def root():
    return {
        "message": "Welcome to ForgeMind AI 🚀",
        "api_v1": "/api/v1",
        "documentation": "/docs",
    }