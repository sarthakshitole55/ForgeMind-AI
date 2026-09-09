from pathlib import Path
from app.core.logger import logger
from app.api.v1 import documents
from langchain_core.messages import HumanMessage,SystemMessage
from app.rag.retriverer.retriverer_service import RetrivererService
from app.services.llm_services import LLMService
from app.prompts.rag import RAG_SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate

class RAGService:
    def __init__(
        self,
        retriverer: RetrivererService | None = None,
        llm: LLMService | None = None,
    ):
        self.retriverer = retriverer or RetrivererService()
        self.llm = llm or LLMService()

    @staticmethod
    def format_chunk(doc) -> str:
        """Formats a single retrieved document chunk with standard source marker."""
        filename = doc.metadata.get("filename")
        if not filename:
            source = doc.metadata.get("source", "Unknown")
            filename = Path(source).name if source != "Unknown" else "Unknown"

        page = doc.metadata.get("page")
        if page is not None and str(page).lower() not in ("none", "unknown"):
            source_marker = f"[Source: {filename} | Page: {page}]"
        else:
            source_marker = f"[Source: {filename}]"

        return f"{source_marker}\n{doc.page_content}"

    def invoke(self, question: str, callbacks: list | None = None):
        documents = self.retriverer.retrive(question)
        
        logger.debug(f"Retrieved {len(documents)} documents for RAG context")
        for i, doc in enumerate(documents, 1):
            logger.debug(f"Document {i} Metadata: {doc.metadata}")

        if not documents:
            logger.info("No relevant documents found above threshold. Returning standard refusal.")
            return {
                "answer": "I couldn't find this information in the uploaded documents.",
                "documents": [],
            }

        context_parts = [self.format_chunk(doc) for doc in documents]
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

        response = self.llm.chat(messages, callbacks=callbacks)

        return {
            "answer": response.content,
            "documents": documents,
        }