from app.core.logger import logger
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
        
        logger.debug(f"Retrieved {len(documents)} documents for RAG context")
        for i, doc in enumerate(documents, 1):
            logger.debug(f"Document {i} Metadata: {doc.metadata}")

        context_parts = []
        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            filename = source.split("/")[-1] if "/" in source else source
            page = doc.metadata.get("page", "Unknown")
            
            context_parts.append(
                f"--------------------------------\n"
                f"Document:\n{filename}\n"
                f"Page: {page}\n\n"
                f"{doc.page_content}"
            )
            
        context = "\n\n".join(context_parts)

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