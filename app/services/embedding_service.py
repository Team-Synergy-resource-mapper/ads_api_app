import os
import numpy as np
import tensorflow as tf
from app.models.siamese_model import load_siamese_branch, load_siamese_model
from app.models.labse_embedding_model import LaBSEEmbedding
# from app.models.vector_db import VectorDB
from app.config import MODEL_PATH, INDEX_PATH, METADATA_PATH, VECTOR_DB_DIR, LABEL_TO_CATEGORY_MAPPING_MAIN, LABEL_TO_CATEGORY_MAPPING_SUB
from app.models import schemas
from typing import List
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating and managing embeddings"""

    def __init__(self):
        """Initialize the embedding service"""
        self.siamese_model = None
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
            # self.embedding_model = load_siamese_branch(MODEL_PATH)
            self.siamese_model = load_siamese_model(MODEL_PATH)
            logger.info(f"Loaded embedding model from {MODEL_PATH}")
            return True
        except Exception as e:
            logger.critical(f"Error loading embedding model: {e}")
            return False

    def _load_labse_model(self):
        """Initialize the LaBSE model"""
        try:
            self.labse_model = LaBSEEmbedding()
            return True
        except Exception as e:
            logger.critical(f"Error loading LaBSE model: {e}")
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

    def generate_ad_embeddings(self, ads: List[schemas.Ad]):

        # Extract ad texts and data
        ad_texts = [ad.text for ad in ads]
        #encode main categories
        main_categories = [LABEL_TO_CATEGORY_MAPPING_MAIN.get(ad.main_category.lower(), 0) for ad in ads]
        #encode subcategories
        sub_categories = [LABEL_TO_CATEGORY_MAPPING_SUB.get(
            ad.sub_category.lower(), 0) for ad in ads]
        
        logger.info(f"Processing {len(ads)} ads")

        # Step 1: Generate LaBSE embeddings
        labse_embeddings = self.labse_model.generate_embeddings(ad_texts)
        logger.info(
            f"Generated LaBSE embeddings with shape: {labse_embeddings.shape}")

        siamese_branch = self.siamese_model.get_layer("siamese_branch")
        main_cat_embedding_layer = self.siamese_model.get_layer("cat1_embedding")
        sub_cat_embedding_layer = self.siamese_model.get_layer("cat2_embedding")

        # Convert categories to numpy arrays
        main_cat_array = np.array(main_categories).reshape(-1, 1)
        sub_cat_array = np.array(sub_categories).reshape(-1, 1)

        # Get category embeddings
        main_cat_embeddings = main_cat_embedding_layer(main_cat_array)
        sub_cat_embeddings = sub_cat_embedding_layer(sub_cat_array)

        # Flatten embeddings
        batch_size = len(ad_texts)
        main_cat_embeddings = tf.reshape(main_cat_embeddings, [batch_size, -1])
        sub_cat_embeddings = tf.reshape(sub_cat_embeddings, [batch_size, -1])

        # Combine text embeddings with category embeddings
        combined_embeddings = tf.concat([labse_embeddings, main_cat_embeddings, sub_cat_embeddings], axis=1)

        ads_embeddings = siamese_branch(combined_embeddings)
        
        logger.info(
            f"Generated ads embeddings with shape: {ads_embeddings.shape}"
        )

        return ads_embeddings

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
