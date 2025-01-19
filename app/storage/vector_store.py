from typing import List, Dict
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, url: str):
        self.client = QdrantClient(url=url)
        self.collection_name = "listing_embeddings"
        self.vector_size = 384  # Adjust based on your embedding size

    async def initialize(self):
        """Initialize vector store"""
        try:
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
        except Exception as e:
            # Collection might already exist
            logger.debug(f"Collection initialization: {str(e)}")
            pass

    async def cleanup(self):
        """Cleanup connections"""
        self.client.close()

    async def store_batch(self, records: List[Dict]) -> List[str]:
        """
        Store batch of embeddings and return vector IDs
        records: List of dicts with 'raw_listing_id', 'embedding', and 'metadata'
        """
        points = []
        vector_ids = []

        for i, record in enumerate(records):
            vector_id = f"v_{record['raw_listing_id']}"
            vector_ids.append(vector_id)
            
            points.append(models.PointStruct(
                id=vector_id,
                vector=record['embedding'],
                payload={
                    'raw_listing_id': record['raw_listing_id'],
                    **record['metadata']
                }
            ))

        # Store in Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return vector_ids

    async def search_similar(
        self, 
        embedding: List[float], 
        limit: int = 10
    ) -> List[Dict]:
        """Search for similar listings"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=limit
        )
        
        return [
            {
                'vector_id': hit.id,
                'score': hit.score,
                'metadata': hit.payload
            }
            for hit in results
        ] 