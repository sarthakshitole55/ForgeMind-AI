from typing import Optional, List
from pydantic import BaseModel, Field
from app.schemas.message import ChatMessage
from app.config.settings import settings


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(
        default_factory=list,
        description="List of conversation messages. Must contain at least one message with non-empty content.",
    )

    temperature: float = Field(
        default=settings.LLM_TEMPERATURE,
        ge=0,
        le=2,
        description="Sampling temperature between 0 and 2",
    )

    max_tokens: int = Field(
        default=settings.LLM_MAX_TOKENS,
        ge=1,
        description="Maximum number of tokens to generate in response",
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Generated answer from ForgeMind AI")
    route: Optional[str] = Field(None, description="Router decision route (rag, web, or maintenance)")
    question: Optional[str] = Field(None, description="The processed question")
    context: Optional[str] = Field(None, description="Retrieved context string if applicable")