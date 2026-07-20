from typing import TypedDict

class AgentState(TypedDict):
    question:str
    route:str
    context:str
    answer:str