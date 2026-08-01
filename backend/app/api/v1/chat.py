from fastapi import APIRouter

from app.agents.graph import graph
from app.schemas.chat import ChatRequest

router = APIRouter()


@router.post("/chat")
def chat(request: ChatRequest):

    result = graph.invoke(
        {
            "question": request.messages[-1].content
        }
    )

    return result