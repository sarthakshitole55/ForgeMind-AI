from app.api.v1 import documents
from langchain_core.messages import HumanMessage,SystemMessage
from app.rag.retriverer.retriverer_service import RetrivererService
from app.services.llm_services import LLMService
from app.prompts.rag import RAG_SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate

class RAGService:
    def __init__(self):
        self.retriverer = RetrivererService()
        self.llm = LLMService()

    def invoke(self, question: str):
        documents = self.retriverer.retrive(question)

        print("\n" + "=" * 70)
        print("RETRIEVED DOCUMENTS")
        print("=" * 70)

        for i, doc in enumerate(documents, 1):
            print(f"\nDocument {i}")
            print("Metadata:", doc.metadata)
            print()
            print(doc.page_content[:500])
            print("-" * 70)

        context = "\n\n".join(
            doc.page_content for doc in documents
        )

        prompt = ChatPromptTemplate.from_messages(
            [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

        messages = prompt.format_messages(
            context=context,
            question=question,
        )

        response = self.llm.chat(messages)

        return {
            "answer": response.content,
            "documents": documents,
        }