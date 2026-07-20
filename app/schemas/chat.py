from pydantic import BaseModel,Field
from app.schemas.message import ChatMessage

class ChatRequest(BaseModel):
    messages:list[ChatMessage]

    temperature:float = Field(
        default=0.2,
        ge=0,
        le=2
    )

    max_tokens:int = 512