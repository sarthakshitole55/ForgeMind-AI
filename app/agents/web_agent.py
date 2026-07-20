from langchain_core.messages import SystemMessage,HumanMessage
from app.services.llm_services import LLMService
from app.services.search_service import SearchService

class WebAgent:
    def __init__(self):
        self.search = SearchService()
        self.llm = LLMService()

    def invoke(self,question:str):
        results = self.search.search(question)

        context = "\n\n".join([result["content"] for result in results])

        response = self.llm.chat([
            SystemMessage(
                content=f"""
                You are an expert AI Research Agent.

                You are given a query and search results.
                Your task is to synthesise this information and give a clear, concise, and comprehensive answer.

                Follow these rules:
                - Use only the provided context.
                - If the answer is not in the context, simply say "I don't know".
                - Do not invent facts.
                - Answer in a neutral, objective tone.
                - Keep the answer professional and easy to understand.

                Query:
                {question}

                Search Results:
                {context}

                Based on the search results, answer the query.
                """
                
            ),
            HumanMessage(content=question),
        ])

        return response.content


web = WebAgent()


# This is what LangGraph calls
def web_node(state):

    state["answer"] = web.invoke(
        state["question"]
    )

    return state