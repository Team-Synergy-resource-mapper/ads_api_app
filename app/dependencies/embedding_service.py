import logging
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

try:
    _embedding_service_instance = EmbeddingService()
    logger.info("EmbeddingService initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize EmbeddingService: {e}")
    _embedding_service_instance = None  # Prevent usage of a broken instance


def get_embedding_service() -> EmbeddingService:
    """Get the already initialized instance of EmbeddingService."""
    if _embedding_service_instance is None:
        raise RuntimeError(
            "EmbeddingService failed to initialize. Check logs for details.")
    return _embedding_service_instance
