import os
from dotenv import load_dotenv
import pathlib

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()

# Model configurations
MODEL_PATH = os.getenv(

    "MODEL_PATH", "./ml_models/matching/siamese_branch_model_labse.keras")

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "vector_db")
INDEX_PATH = os.path.join(VECTOR_DB_DIR, "hnsw_index.bin")
METADATA_PATH = os.path.join(VECTOR_DB_DIR, "metadata.pkl")

# LaBSE model configuration
LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
BATCH_SIZE = 32

# Server configurations
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# Vector DB configurations
VECTOR_DIM = 768  # Default dimension for embeddings
MAX_ELEMENTS = 100000  # Maximum number of elements in vector DB
EF_CONSTRUCTION = 200  # HNSW index parameter
M = 16  # HNSW index parameter
EF_SEARCH = 50  # HNSW search parameter


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
    16: 'lorry truck',
    15: 'land',
    14: 'house',
    13: 'commercial property',
    12: 'car',
    11: 'bike',
    10: 'bicycle',
    9: 'apartment',
    8: 'tvs',
    7: 'mobile phones & tablets',
    6: 'mobile phone accessories',
    5: 'electronic home appliances',
    4: 'computers',
    3: 'computer accessories',
    2: 'cameras & camcorders',
    1: 'audio & mp3',
    0: 'air conditions & electrical fittings'
}
