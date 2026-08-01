from app.rag.retriverer.retriverer_service import RetrivererService
retriverer = RetrivererService()

result = retriverer.retrive(
    query="What is Deep Learning?",
    k=5
)

print(f"Retrived {len(result)} documents")

for i,doc in enumerate(result,start=1):
    print("=" * 70)
    print(f"Result {i}")
    print(f"Page : {doc.metadata.get('page')}")
    print(f"Source : {doc.metadata.get('source')}")
    print()
    print(doc.page_content[:400])
    print()