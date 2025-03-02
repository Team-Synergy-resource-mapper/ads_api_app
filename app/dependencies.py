from app.services.embedding_service import EmbeddingService

# Initialize singleton services
embedding_service = EmbeddingService()
embedding_service.initialize()


def get_embedding_service():
    """Dependency to get the embedding service"""
    return embedding_service
