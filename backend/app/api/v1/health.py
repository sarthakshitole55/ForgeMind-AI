from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field
from app.config.settings import settings
from app.core.logger import logger

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    project: str = Field(default="ForgeMind AI", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")


class ReadinessDetails(BaseModel):
    storage: str = Field(..., description="Local document storage availability status")
    vector_store: str = Field(..., description="Local vector store availability status")


class ReadinessResponse(BaseModel):
    status: str = Field(..., description="Readiness status ('ready' or 'not_ready')")
    details: ReadinessDetails = Field(..., description="Component readiness details")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application liveness check",
    description="Returns simple OK status indicating the process is running.",
)
@router.get(
    "/health/live",
    response_model=HealthResponse,
    summary="Container liveness probe",
    description="Confirms application process is alive for container orchestration.",
)
def liveness():
    return HealthResponse(
        status="ok",
        project=settings.APP_NAME,
        version=settings.APP_VERSION,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Application readiness probe",
    description="Verifies local storage and vector store availability before routing user traffic.",
    responses={
        200: {"description": "All components are ready to serve requests", "model": ReadinessResponse},
        503: {"description": "One or more critical components are unavailable", "model": ReadinessResponse},
    },
)
def readiness(response: Response):
    details = {}
    is_ready = True

    # 1. Local storage check
    try:
        data_dir = settings.get_data_dir()
        if data_dir.exists():
            details["storage"] = "ready"
        else:
            details["storage"] = "unavailable"
            is_ready = False
    except Exception as e:
        logger.warning(f"Storage readiness check failed: {e}")
        details["storage"] = "unavailable"
        is_ready = False

    # 2. Vector store check (lightweight local metadata check, zero external network calls)
    try:
        from app.rag.vectorstore.chroma_store import ChromaVectorStore
        store = ChromaVectorStore()
        # Fast local sqlite metadata count
        _ = store.db._collection.count()
        details["vector_store"] = "ready"
    except Exception as e:
        logger.warning(f"Vector store readiness check failed: {e}")
        details["vector_store"] = "unavailable"
        is_ready = False

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="not_ready",
            details=ReadinessDetails(**details),
        )

    return ReadinessResponse(
        status="ready",
        details=ReadinessDetails(**details),
    )