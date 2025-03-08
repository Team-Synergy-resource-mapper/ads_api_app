import logging
from app.models.labse_model import LaBSEEmbedding


logger = logging.getLogger(__name__)
_labse_instance = None


def get_labse_model() -> LaBSEEmbedding:
    """Get the singleton instance of LaBSEEmbedding with error handling"""
    global _labse_instance
    if _labse_instance is None:
        try:
            _labse_instance = LaBSEEmbedding()
        except RuntimeError as e:
            logger.error(f"Singleton instantiation failed: {e}")
            raise e  # Propagate the exception
    return _labse_instance
