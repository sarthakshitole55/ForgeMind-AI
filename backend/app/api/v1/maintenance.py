from fastapi import APIRouter, Request

from app.core.logger import logger
from app.core.middleware import get_request_id
from app.core.observability import ObservabilityService
from app.schemas.maintenance import (
    ActionExecutionResult,
    MaintenanceActionRequest,
    MaintenanceReport,
)
from app.services.maintenance_service import MaintenanceService

router = APIRouter(tags=["Maintenance"])

_service = MaintenanceService()


@router.get(
    "/maintenance/report",
    response_model=MaintenanceReport,
    summary="System health and consistency report",
    description="Returns deterministic inspection findings for storage, vector store, and document consistency.",
    responses={
        200: {"description": "Complete system maintenance report", "model": MaintenanceReport},
        500: {"description": "Internal error generating maintenance report"},
    },
)
def get_maintenance_report(req: Request) -> MaintenanceReport:
    request_id = get_request_id(req)
    logger.info(f"Generating maintenance report | Request ID: {request_id}")

    report = _service.inspect_system_and_consistency()
    return report


@router.post(
    "/maintenance/execute",
    response_model=ActionExecutionResult,
    summary="Execute safe maintenance action",
    description="Executes an allowlisted maintenance action (e.g. REINDEX_DOCUMENT, REMOVE_ORPHAN_VECTOR).",
    responses={
        200: {"description": "Action executed and verified", "model": ActionExecutionResult},
        400: {"description": "Invalid action, target, or path traversal attempt"},
        500: {"description": "Execution failure"},
    },
)
def execute_maintenance_action(action_req: MaintenanceActionRequest, req: Request) -> ActionExecutionResult:
    request_id = get_request_id(req)
    logger.info(
        f"Executing maintenance action '{action_req.action.value}' on '{action_req.target}' | Request ID: {request_id}"
    )

    with ObservabilityService.start_request_trace(
        name="maintenance_action",
        user_query=f"{action_req.action.value} {action_req.target}",
        metadata={"request_id": request_id, "action": action_req.action.value, "target": action_req.target},
    ):
        result = _service.execute_action(action_req.action, action_req.target)
        return result
