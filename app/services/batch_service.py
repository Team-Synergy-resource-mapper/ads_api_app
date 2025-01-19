from typing import List, Dict, Tuple
import logging
from datetime import datetime

from ..models.schemas import RawListing, ProcessedListing
from ..storage.vector_store import VectorStore
from ..storage.postgres import PostgresClient
from ..core.ad_classifier import AdClassifier
from ..config.settings import BATCH_SIZE, MODEL_VERSION

logger = logging.getLogger(__name__)

class BatchProcessingService:
    """
    Main service that coordinates between database, vector store, and classification
    """
    def __init__(
        self,
        postgres_client: PostgresClient,
        vector_store: VectorStore,
        classifier: AdClassifier,
        batch_size: int = BATCH_SIZE
    ):
        self.postgres = postgres_client
        self.vector_store = vector_store
        self.classifier = classifier
        self.batch_size = batch_size

    async def process_batch(self) -> Tuple[int, int]:
        """
        Process a single batch of unprocessed listings
        Returns: (processed_count, error_count)
        """
        try:
            # 1. Fetch unprocessed listings
            raw_listings = await self.postgres.fetch_unprocessed_listings(
                limit=self.batch_size
            )

            if not raw_listings:
                logger.info("No unprocessed listings found")
                return 0, 0

            # 2. Prepare texts for processing
            texts = [self._prepare_listing_text(listing) for listing in raw_listings]

            # 3. Generate embeddings and classify
            embeddings = self.classifier.generate_embeddings(texts)
            classifications = self.classifier.classify_batch(embeddings)

            # 4. Store embeddings in vector store
            vector_ids = await self._store_embeddings(
                raw_listings, embeddings, classifications
            )

            # 5. Create processed listings
            processed_listings = self._create_processed_listings(
                raw_listings, classifications, vector_ids
            )

            # 6. Store processed listings and update raw listings
            await self._store_results(raw_listings, processed_listings)

            return len(raw_listings), 0

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}", exc_info=True)
            return 0, len(raw_listings) if raw_listings else 0

    async def _store_embeddings(
        self, 
        listings: List[RawListing], 
        embeddings: List[List[float]], 
        classifications: List[Dict]
    ) -> List[str]:
        """Store embeddings in vector store and return vector IDs"""
        vector_records = [
            {
                'raw_listing_id': listing.id,
                'embedding': embedding,
                'metadata': {
                    'main_category': classification['main_category'],
                    'sub_category': classification['sub_category'],
                    'title': listing.title
                }
            }
            for listing, embedding, classification in zip(
                listings, embeddings, classifications
            )
        ]
        
        return await self.vector_store.store_batch(vector_records)

    def _create_processed_listings(
        self,
        raw_listings: List[RawListing],
        classifications: List[Dict],
        vector_ids: List[str]
    ) -> List[ProcessedListing]:
        """Create ProcessedListing objects from results"""
        return [
            ProcessedListing(
                raw_listing_id=raw.id,
                main_category=classification['main_category'],
                sub_category=classification['sub_category'],
                leaf_category=classification.get('leaf_category'),
                vector_id=vector_id,
                model_version=MODEL_VERSION,
                classification_metadata=classification
            )
            for raw, classification, vector_id in zip(
                raw_listings, classifications, vector_ids
            )
        ]

    async def _store_results(
        self,
        raw_listings: List[RawListing],
        processed_listings: List[ProcessedListing]
    ):
        """Store results in postgres and mark raw listings as processed"""
        async with self.postgres.transaction():
            # Store processed listings
            await self.postgres.store_processed_listings(processed_listings)
            
            # Mark raw listings as processed
            raw_ids = [listing.id for listing in raw_listings]
            await self.postgres.mark_listings_processed(raw_ids)

    @staticmethod
    def _prepare_listing_text(listing: RawListing) -> str:
        """Prepare listing text for classification"""
        components = [
            listing.title,
            listing.description,
            listing.meta_data,
            ' '.join(listing.attributes or []),
            ' '.join(listing.breadcrumbs or [])
        ]
        return ' '.join(filter(None, components)) 