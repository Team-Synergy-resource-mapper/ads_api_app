import logging
from app.models.siamese_model import load_siamese_model
from app.config.config import SIAMESE_MODEL_PATH

logger = logging.getLogger(__name__)

_siamese_model_instance = None  


def get_siamese_model():
    """Get the singleton instance of the Siamese model."""
    global _siamese_model_instance
    if _siamese_model_instance is None:
        try:
            _siamese_model_instance = load_siamese_model(SIAMESE_MODEL_PATH)
            logger.info("Siamese model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Siamese model: {e}")
            raise RuntimeError(
                "Could not load the Siamese model. Check logs for details.")
    return _siamese_model_instance
