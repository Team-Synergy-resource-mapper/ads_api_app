import numpy as np
from sentence_transformers import SentenceTransformer
from app.config.config import LABSE_MODEL_NAME, BATCH_SIZE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaBSEEmbedding:
    """Singleton class for handling LaBSE embeddings with error handling"""

    _instance = None  # Class-level variable to store the singleton instance

    def __new__(cls, model_name=LABSE_MODEL_NAME):
        """Ensure only one instance of LaBSEEmbedding exists, with error handling"""
        if cls._instance is None:
            try:
                cls._instance = super(LaBSEEmbedding, cls).__new__(cls)
                cls._instance._initialize(model_name)
            except Exception as e:
                cls._instance = None  # Reset instance if initialization fails
                logger.error(f"Failed to initialize LaBSE model: {e}")
                raise RuntimeError(
                    "Failed to load LaBSE model. Check logs for details.")
        return cls._instance

    def _initialize(self, model_name):
        """Initialize the LaBSE model"""
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info(f"Successfully loaded LaBSE model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading LaBSE model: {e}")
            raise RuntimeError(f"Error loading LaBSE model: {e}")

    def generate_embeddings(self, sentences, batch_size=BATCH_SIZE):
        """
        Generate LaBSE embeddings for sentences with error handling
        
        Args:
            sentences (list): List of sentences to encode
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: NumPy array of embeddings (len(sentences), embedding_dim)
        """
        if not isinstance(sentences, list) or not sentences:
            logger.error(
                "Invalid input: sentences must be a non-empty list of strings.")
            raise ValueError(
                "Input sentences must be a non-empty list of strings.")

        logger.info(
            f"Generating LaBSE embeddings for {len(sentences)} sentences")

        all_embeddings = []
        try:
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i + batch_size]

                # Check if batch contains valid text
                if not all(isinstance(s, str) and s.strip() for s in batch_sentences):
                    logger.error(
                        "Invalid sentence detected in batch. Ensure all are non-empty strings.")
                    raise ValueError(
                        "All sentences must be non-empty strings.")

                batch_embeddings = self.model.encode(
                    batch_sentences, convert_to_numpy=True)
                all_embeddings.append(batch_embeddings)

                logger.info(
                    f"Processed batch {i // batch_size + 1}/{(len(sentences) + batch_size - 1) // batch_size}")

            return np.vstack(all_embeddings)

        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")
