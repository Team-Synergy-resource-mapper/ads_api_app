from app.db.vector_db import VectorDB
from app.config.config import MONGODB_URI, DB_NAME, COLLECTION_NAME


vector_db_instance = VectorDB(
  mongodb_uri=MONGODB_URI,
  db_name=DB_NAME,
  collection_name=COLLECTION_NAME,
)

def get_vector_db() -> VectorDB:
  return vector_db_instance