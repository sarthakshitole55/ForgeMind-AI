from app.rag.rag_service import RAGService

rag = RAGService()

response = rag.ask(
    "What is competetive programming ?"
)
print("\nAnswer :\n")
print(response["answer"])

print("\nSources :\n")
for doc in response["documents"]:
    print(
        f"Page {doc.metadata.get('page')} | "
        f"{doc.metadata.get('source')}"
    )