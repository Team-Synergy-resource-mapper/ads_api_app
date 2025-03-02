import os
import numpy as np
import tensorflow as tf
from ..models.tensorflow_models import load_siamese_branch
from app.models.embedding_models import LaBSEEmbedding
# from app.models.vector_db import VectorDB
from app.config.config import MODEL_PATH, INDEX_PATH, METADATA_PATH, VECTOR_DB_DIR


class EmbeddingService:
    """Service for generating and managing embeddings"""

    def __init__(self):
        """Initialize the embedding service"""
        self.embedding_model = None
        self.labse_model = None
        # self.vector_db = None

        # Make sure the vector database directory exists
        # os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    def initialize(self):
        """Initialize models and vector database"""
        self._load_embedding_model()
        self._load_labse_model()
        # self._load_vector_db()

    def _load_embedding_model(self):
        """Load the embedding model"""
        try:
            self.embedding_model = load_siamese_branch(MODEL_PATH)
            print(f"Loaded embedding model from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            return False

    def _load_labse_model(self):
        """Initialize the LaBSE model"""
        try:
            self.labse_model = LaBSEEmbedding()
            return True
        except Exception as e:
            print(f"Error loading LaBSE model: {e}")
            return False

    # def _load_vector_db(self):
    #     """Initialize or load the vector database"""
    #     try:
    #         if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
    #             self.vector_db = VectorDB.load(INDEX_PATH, METADATA_PATH)
    #         else:
    #             self.vector_db = VectorDB()
    #             print("Initialized new vector database")
    #         return True
    #     except Exception as e:
    #         print(f"Error initializing vector database: {e}")
    #         self.vector_db = VectorDB()
    #         return False

    def generate_ad_embeddings(self, ads):
        """
        Generate embeddings for a list of ads
        
        Args:
            ads: List of Ad objects
            
        Returns:
            Dictionary with LaBSE embeddings, final embeddings, and ad data
        """
        # Extract ad texts and data
        ad_texts = []
        # ad_data = []

        for ad in ads:
            # Combine title and description for embedding
            # combined_text = f"{ad.title} {ad.description}"
            combined_text = ad.description
            ad_texts.append(combined_text)

            # Store ad details
            # ad_data.append({
            #     "title": ad.title,
            #     "description": ad.description,
            #     "metadata": ad.metadata or {}
            # })

        print(f"Processing {len(ad_texts)} ads")

        # Step 1: Generate LaBSE embeddings
        labse_embeddings = self.labse_model.generate_embeddings(ad_texts)
        print(
            f"Generated LaBSE embeddings with shape: {labse_embeddings.shape}")

        # Optional: Store in vector database for later retrieval
        # if self.vector_db is not None:
        #     ids = self.vector_db.add_ads(final_embeddings, ad_data)
        #     self.vector_db.save(INDEX_PATH, METADATA_PATH)
        #     print(f"Stored embeddings in vector database with IDs: {ids}")

        # Return both embeddings

        # Use embedding model to process labse embeddings

        ads_embeddings = self.embedding_model.predict(labse_embeddings)
        print(
            f"Generated ads embeddings with shape: {ads_embeddings.shape}"
        )

        return {
            "message": f"Successfully generated embeddings for {len(ad_texts)} ads",
            "ads_embeddings" : ads_embeddings
        }

    # def search_similar_ads(self, query, top_k=10):
    #     """
    #     Search for similar ads based on a query string
        
    #     Args:
    #         query: Search query string
    #         top_k: Number of results to return
            
    #     Returns:
    #         List of search results
    #     """
    #     # Generate LaBSE embedding for query
    #     query_labse_embedding = self.labse_model.generate_embeddings([query])[
    #         0]

    #     # Process through embedding model
    #     query_labse_embedding_tf = tf.convert_to_tensor(
    #         [query_labse_embedding], dtype=tf.float32)
    #     query_final_embedding = self.embedding_model(
    #         query_labse_embedding_tf).numpy()[0]

    #     # Search in vector database
    #     results = self.vector_db.search(query_final_embedding, k=top_k)

    #     return results

    # def get_health(self):
    #     """
    #     Get health status of the service
        
    #     Returns:
    #         Dictionary with health status
    #     """
    #     return {
    #         "embedding_model": self.embedding_model is not None,
    #         "labse_model": self.labse_model is not None,
    #         "vector_db": self.vector_db is not None,
    #         "ad_count": self.vector_db.current_id if self.vector_db else 0
    #     }
