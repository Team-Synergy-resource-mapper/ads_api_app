from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import List
import app.models.schemas as schemas
import logging
from bson.binary import BinaryVectorDtype, Binary
from app.models.schemas import Ad

logger = logging.getLogger(__name__)

class VectorDB:
  def __init__(self, mongodb_uri: str, db_name: str, collection_name: str):
      self.client = MongoClient(mongodb_uri)
      self.db = self.client[db_name]
      self.collection = self.db[collection_name]
      self._ensure_index_exists()

  def _ensure_index_exists(self):
      index_name  = "vectors_index"
      indexes = list(self.collection.list_search_indexes())
      index_exists = any(idx["name"] == index_name for idx in indexes)

      if not index_exists:
          search_index_model = SearchIndexModel(
              definition={
                  "fields": [
                      {
                          "type": "vector",
                          "path": "ad_embedding",
                          "numDimensions": 1536,
                          "similarity": "cosine",
                          # "quantization": "scalar"
                      }
                  ]
              },
              name= index_name,
              type="vectorSearch",
          )
          
          self.collection.create_search_index(model = search_index_model)
          logger.info("Vector search index created.")

  def _generate_bson_vector(self, vector, vector_dtype = BinaryVectorDtype.FLOAT32):
      return Binary.from_vector(vector, vector_dtype)
  def insert_ad(self, ad : Ad , embedding):
      
      embedding = self._generate_bson_vector(embedding)
      return self.collection.insert_one(
          {
              **ad.model_dump(),
              "ad_embedding": embedding,
              "created_at": datetime.now()
          }
      )

  def insert_ads(self, ads : List[Ad], embeddings):
      if len(ads) != len(embeddings):
          logger.error("Ads and embeddings lengths do not match")
          raise ValueError("Ads and embeddings lengths do not match")
      documents = [
          {
              
          }
          for i, ad in enumerate(ads)
      ]

      return self.collection.insert_many(documents)
  
  def search_similar(self, query_embedding: np.ndarray, limit: int = 5):
        pipeline = [
            {"$search": {
                "index": "vector_index",
                "knnBeta": {
                    "vector": query_embedding.tolist(),
                    "path": "embedding",
                    "k": limit
                }
            }},
            {"$project": {
                "text": 1,
                "main_category": 1,
                "sub_category": 1,
                "transaction_type": 1,
                "wanted_offering": 1,
                "score": {"$meta": "searchScore"},
                "_id": 1
            }}
        ]
        return list(self.collection.aggregate(pipeline))

