import numpy as np
import tensorflow as tf
from app.config.config import  LABEL_TO_CATEGORY_MAPPING_MAIN, LABEL_TO_CATEGORY_MAPPING_SUB
from app.models import schemas
from typing import List
import logging
from app.dependencies.siamese_model import get_siamese_model
from app.dependencies.labse_model import get_labse_model

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating and managing embeddings"""

    def __init__(self):
        """Initialize the embedding service"""
        self.siamese_model = get_siamese_model()
        self.labse_model = get_labse_model()
    

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


