import re
import uuid
from contextvars import ContextVar
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

SAFE_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id(request: Request | None = None) -> str:
    """
    Returns the sanitized correlation request ID from request.state or contextvar.
    """
    if request is not None and hasattr(request, "state") and hasattr(request.state, "request_id"):
        return request.state.request_id
    return request_id_ctx.get() or ""


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that generates or propagates a correlation X-Request-ID.
    - Sanitizes and validates incoming IDs (rejecting oversized/malformed headers).
    - Injects request ID into request.state and contextvars for logging/tracing.
    - Attaches X-Request-ID to all HTTP response headers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming_id = request.headers.get("X-Request-ID")
        if incoming_id and SAFE_REQUEST_ID_REGEX.match(incoming_id):
            request_id = incoming_id
        else:
            request_id = str(uuid.uuid4())

        request.state.request_id = request_id
        token = request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_ctx.reset(token)
