from pydantic import BaseModel,Field
from app.schemas.message import ChatMessage
from app.config.settings import settings

class ChatRequest(BaseModel):
    messages:list[ChatMessage]

    temperature:float = Field(
        default=settings.LLM_TEMPERATURE,
        ge=0,
        le=2
    )

    max_tokens:int = settings.LLM_MAX_TOKENS