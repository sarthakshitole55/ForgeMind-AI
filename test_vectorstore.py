from pathlib import Path

from app.rag.indexer import DocumentIndexer

indexer = DocumentIndexer()

for pdf in Path("data").glob("*.pdf"):
    print("=" * 60)
    print(f"Indexing: {pdf.name}")

    result = indexer.index_pdf(str(pdf))

    print(result)

print("\n✅ All documents indexed!")