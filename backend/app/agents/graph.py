from langgraph.graph import StateGraph,END
from app.agents.state import AgentState
from app.agents.supervisor import supervisor
from app.agents.rag_agent import rag_node
from app.agents.web_agent import web_node
from app.agents.maintenance_agent import maintenance_node

builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor)
builder.add_node("rag", rag_node)
builder.add_node("web", web_node)
builder.add_node("maintenance", maintenance_node)

builder.set_entry_point("supervisor")

def route(state):

    return state["route"]


builder.add_conditional_edges(
    "supervisor",
    route,
    {
        "rag": "rag",
        "web": "web",
        "maintenance": "maintenance",
    },
)

builder.add_edge("rag", END)
builder.add_edge("web", END)
builder.add_edge("maintenance", END)

graph = builder.compile()