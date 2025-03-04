import os
from dotenv import load_dotenv
import pathlib

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()

# Model configurations
MODEL_PATH = os.getenv(
    "MODEL_PATH", BASE_DIR / "ml_models/matching/siamese_branch_model_labse.keras")
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