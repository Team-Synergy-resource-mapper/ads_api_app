import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import LABSE_MODEL_NAME, BATCH_SIZE


class LaBSEEmbedding:
    """Class for handling LaBSE embeddings"""

    def __init__(self, model_name=LABSE_MODEL_NAME):
        """Initialize the LaBSE model"""
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"Loaded LaBSE model: {model_name}")

    def generate_embeddings(self, sentences, batch_size=BATCH_SIZE):
        """
        Generate LaBSE embeddings for sentences
        
        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for processing
            
        Returns:
            NumPy array of embeddings with shape (len(sentences), embedding_dim)
        """
        print(f"Generating LaBSE embeddings for {len(sentences)} sentences")

        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_sentences, convert_to_numpy=True)
            all_embeddings.append(batch_embeddings)
            print(f"Processed batch {i // batch_size + 1}/{(len(sentences) + batch_size - 1) // batch_size}")

        # Concatenate all batch embeddings into a single NumPy array
        return np.vstack(all_embeddings)
