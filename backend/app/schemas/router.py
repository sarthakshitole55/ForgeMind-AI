from typing import Literal

from pydantic import BaseModel


class RouteDecision(BaseModel):
    """
    Output returned by the supervisor LLM.
    """

    route: Literal[
        "rag",
        "web",
        "maintenance",
    ]

    reason: str