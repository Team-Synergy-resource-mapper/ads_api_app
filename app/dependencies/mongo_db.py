import logging
from app.db.vector_db import VectorDB
from app.config.config import MONGODB_URI, DB_NAME, COLLECTION_NAME

logger = logging.getLogger(__name__)

try:
    _vector_db_instance = VectorDB(
        mongodb_uri=MONGODB_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
    )
    logger.info("VectorDB connection established successfully.")
except Exception as e:
    logger.critical(f"Failed to connect to VectorDB: {e}")
    _vector_db_instance = None  


def get_vector_db() -> VectorDB:
    """Return the already initialized VectorDB instance."""
    if _vector_db_instance is None:
        raise RuntimeError(
            "VectorDB failed to initialize. Check logs for details.")
    return _vector_db_instance
