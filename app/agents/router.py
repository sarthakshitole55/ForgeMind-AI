from langchain_core.messages import HumanMessage, SystemMessage

from app.prompts.router import ROUTER_PROMPT
from app.schemas.router import RouteDecision
from app.services.llm_services import LLMService


class RouterAgent:

    def __init__(self):
        self.llm = LLMService()

    def route(self, question: str) -> RouteDecision:

        structured_llm = (
            self.llm
            .get_llm()
            .with_structured_output(RouteDecision)
        )

        return structured_llm.invoke(
            [
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(content=question),
            ]
        )