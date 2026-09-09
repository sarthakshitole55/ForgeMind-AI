from fastapi import APIRouter, Request

from app.agents.graph import graph
from app.schemas.chat import ChatRequest, ChatResponse
from app.core.observability import ObservabilityService
from app.core.exceptions import ForgeMindError, ValidationError
from app.core.logger import logger
from app.core.middleware import get_request_id

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send chat message",
    description="Submits user message to the ForgeMind reasoning graph and returns the AI response.",
    responses={
        200: {"description": "Successful chat response", "model": ChatResponse},
        400: {"description": "Invalid or empty message payload"},
        422: {"description": "Request schema validation failure"},
        502: {"description": "AI model provider failure"},
        503: {"description": "Vector store or dependency unavailable"},
        504: {"description": "AI model provider request timeout"},
        500: {"description": "Internal server error"},
    },
)
def chat(request: ChatRequest, req: Request):
    if not request.messages:
        logger.warning("Chat request rejected: messages list is empty")
        raise ValidationError("Request must contain at least one message")

    last_message = request.messages[-1]
    question = (last_message.content or "").strip()
    if not question:
        logger.warning("Chat request rejected: message content is empty")
        raise ValidationError("Message content cannot be empty")

    request_id = get_request_id(req)

    with ObservabilityService.start_request_trace(
        name="chat_request",
        user_query=question,
        metadata={"request_id": request_id} if request_id else None,
    ) as (_trace_id, handler):
        config = {"callbacks": [handler]} if handler else {}
        try:
            result = graph.invoke(
                {
                    "question": question
                },
                config=config,
            )
            return result
        except ForgeMindError:
            raise
        except Exception as e:
            logger.error(f"Chat workflow failed unexpectedly: {e}", exc_info=True)
            raise ForgeMindError(
                "An unexpected error occurred while processing your request.",
                error_type="InternalError",
                status_code=500,
            )