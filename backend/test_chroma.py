from app.rag.vectorstore.chroma_store import ChromaVectorStore
db = ChromaVectorStore().db
docs = db.max_marginal_relevance_search("test", k=2, fetch_k=5)
print("MMR docs:", len(docs))
docs_scores = db.similarity_search_with_relevance_scores("test", k=5)
print("Scores:", [s for d, s in docs_scores])
