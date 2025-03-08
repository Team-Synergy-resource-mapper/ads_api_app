import os
from dotenv import load_dotenv
import pathlib

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()

# Model configurations
SIAMESE_MODEL_PATH = os.getenv(
    "SIAMESE_MODEL_PATH",
    "./ml_models/matching/siamese_model_with_categories_laBSE.keras")


# LaBSE model configuration
LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
BATCH_SIZE = 32

# Server configurations
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# Vector DB configurations
VECTOR_DIM = 256  # Default dimension for embeddings
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "classified_ads_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ads")


# category mappings siamese model

LABEL_TO_CATEGORY_MAPPING_MAIN = {
    'electronics': 0,
    'property': 1,
    'vehicle': 2
}

LABEL_TO_CATEGORY_MAPPING_SUB = {
    19: 'van',
    18: 'three-wheeler',
    17: 'room & annex',
    16: 'lorry_truck',
    15: 'land',
    14: 'house',
    13: 'commercial property',
    12: 'car',
    11: 'bike',
    10: 'bicycle',
    9:  'apartment',
    8:  'tvs',
    7: 'mobile phones & tablets',
    6: 'mobile phone accessories',
    5: 'electronic home appliances',
    4: 'computers',
    3: 'computer accessories',
    2: 'cameras & camcorders',
    1: 'audio & mp3',
    0: 'air conditions & electrical fittings'
}
