from pymongo import MongoClient
import numpy as np

# MongoDB client setup (replace with your MongoDB URI)
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client['classified_ads_db']
embeddings_collection = db['embeddings']

def save_embeddings_to_mongodb(embeddings):
    """Save embeddings to MongoDB."""
    for embedding in embeddings:
        embeddings_collection.insert_one({'embedding': embedding.tolist()})
