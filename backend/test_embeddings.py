
from app.rag.embeddings.embedding_service import EmbeddingService
embedding_service = EmbeddingService()
embeddings = embedding_service.get_embeddings()

vector = embeddings.embed_query(
    "How often should the hydraulic filter be replaced?"
)
print(f"Vector Length : {len(vector)}")
print(vector[:10])