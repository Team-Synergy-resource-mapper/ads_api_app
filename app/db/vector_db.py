import logging
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import numpy as np
from datetime import datetime
from typing import List
import app.models.schemas as schemas
from bson.binary import BinaryVectorDtype, Binary
from app.models.schemas import Ad
from bson import ObjectId

logger = logging.getLogger(__name__)

class VectorDB:
  def __init__(self, mongodb_uri: str, db_name: str, collection_name: str):
      self.client = MongoClient(mongodb_uri)
      self.db = self.client[db_name]
      self.collection = self.db[collection_name]
      self.ad_embedding_index_name = "vector_index"
      self._ensure_index_exists()
      

  def _ensure_index_exists(self):
      
      indexes = list(self.collection.list_search_indexes())
      index_exists = any(idx["name"] == self.ad_embedding_index_name for idx in indexes)

      if not index_exists:
          search_index_model = SearchIndexModel(
              definition={
                  "fields": [
                      {
                          "type": "vector",
                          "path": "ad_embedding",
                          "numDimensions": 256,
                          "similarity": "cosine",
                          # "quantization": "scalar"
                      },
                      {
                          "type": "filter",
                          "path": "main_category"
                      },
                      {
                          "type": "filter",
                          "path" : "sub_category"
                      },
                      {
                          "type" : "filter",
                          "path": "wanted_offering"
                      }

                  ]
              },
              name= self.ad_embedding_index_name,
              type="vectorSearch",
          )
          
          self.collection.create_search_index(model = search_index_model)
          logger.info("Vector search index created.")

  def _generate_bson_vector(self, vector, vector_dtype = BinaryVectorDtype.FLOAT32):
      
      return Binary.from_vector(vector, vector_dtype)
  def insert_ad(self, ad : Ad , embedding):
      
      embedding = self._generate_bson_vector(embedding)
      doc = {
          **ad.model_dump(),
          "ad_embedding": embedding,
          "created_at": datetime.now()
      }
      result =  self.collection.insert_one(doc)
      logger.info(f"Ad inserted with id: {result.inserted_id}")
      return result

  def insert_ads(self, ads : List[Ad], embeddings):
      if len(ads) != len(embeddings):
          logger.error("Ads and embeddings lengths do not match")
          raise ValueError("Ads and embeddings lengths do not match")
      documents = [
          {
              **ad.model_dump(),
              "ad_embedding": self._generate_bson_vector(embeddings[i]),
              "created_at": datetime.now()
          }
          for i, ad in enumerate(ads)
      ]
      

      result =  self.collection.insert_many(documents)
      logger.info(f"Ads inserted with ids: {', '.join(str(id) for id in result.inserted_ids)}")
      return result
  
  def search_similar_by_query_embedding(self, query_embedding: np.ndarray, limit: int = 10, number_of_candidates: int = 100):
        pipeline = [
            {
                "$vectorSearch": {
                    # "exact": false, defualt : false for ann search
                    # "filter": { },
                    "index": self.ad_embedding_index_name,
                    "limit": limit,
                    "numCandidates": number_of_candidates,
                    "path": "ad_embedding",
                    "queryVector": query_embedding,
                }
            },
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
        logger.info(f"Searching for similar ads using query embedding")
        result =  list(self.collection.aggregate(pipeline))
        logger.info(f"Found {len(result)} similar ads")
        return result
  
  def search_similar_by_ad_id(self,ad_id : ObjectId, limit: int = 10, number_of_candidates: int = 100):
      
      # Fetch the embedding and category details of the given ad_id
      ad = self.collection.find_one(
         {
             '_id': ad_id,
         },
         {
             "ad_embedding" : 1, "main_category": 1, "sub_category": 1, "wanted_offering": 1
         }
      )

      if not ad or "ad_embedding" not in ad:
        logger.error(f"Ad with ID {ad_id} not found or missing embedding.")
        return []
      logger.info(f"Fetched ad with ID {ad_id}")  

          
      query_embedding = ad["ad_embedding"]
      main_category = ad.get("main_category", None)  # Extract main category
      sub_category = ad.get("sub_category", None)  # Extract subcategory
      wanted_offering = ad.get("wanted_offering", None)  # Extract wanted/offering type   
       # Determine the opposite of wanted_offering
      opposite_wanted_offering = "offering" if wanted_offering == "wanted" else "wanted"

      # Construct the filter
       # Construct the filter dynamically
      filter_criteria = {
        #   "_id": {"$ne": ad_id},  # Exclude the original ad
          "wanted_offering": opposite_wanted_offering  # Get opposite type of ad
      }
    
    # Add category filters only if they exist
      if main_category:
          filter_criteria["main_category"] = main_category
      if sub_category:
          filter_criteria["sub_category"] = sub_category

      pipeline = [
          {
              "$vectorSearch": {
                  # "exact": false, defualt : false for ann search
                  "filter": filter_criteria,
                  "index": self.ad_embedding_index_name,
                  "limit": limit,
                  "numCandidates": number_of_candidates,
                  "path": "ad_embedding",
                  "queryVector": query_embedding,
              }
          },
          {"$project": {
              "ad_embedding": 0,
              "score": {"$meta": "vectorSearchScore"},
              # "text": 1,
              # "main_category": 1,
              # "sub_category": 1,
              # "transaction_type": 1,
              # "wanted_offering": 1,
              # "_id": 1
          }}
      ]
      logger.info(f"Searching for similar ads using ad id")
      result = list(self.collection.aggregate(pipeline))
      logger.info(f"Found {len(result)} similar ads")
      return result

