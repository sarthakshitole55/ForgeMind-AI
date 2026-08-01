from typing import Literal

from app.agents.router import RouterAgent
from app.core.logger import logger

from app.agents.state import AgentState

def supervisor(state: AgentState) -> dict[str, str]:

    router = RouterAgent()

    decision = router.route(state["question"])

    logger.info(f"Supervisor Route Decision | Question: '{state['question']}' | Route: {decision.route} | Reason: {decision.reason}")

    return {"route": decision.route}