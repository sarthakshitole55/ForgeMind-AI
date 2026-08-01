from app.rag.rag_service import RAGService

rag = RAGService()

def rag_node(state):
    result = rag.invoke(state["question"])
    state["answer"] = result["answer"]

    return state