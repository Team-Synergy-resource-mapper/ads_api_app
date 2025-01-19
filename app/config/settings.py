from pathlib import Path

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Path to the directory containing the ML models
ML_MODELS_DIR = PROJECT_ROOT / "ml_models"

# Category-specific paths
MAIN_MODELS_DIR = ML_MODELS_DIR / "main"
ELECTRONICS_MODELS_DIR = ML_MODELS_DIR / "electronics"
PROPERTY_MODELS_DIR = ML_MODELS_DIR / "property"
VEHICLES_MODELS_DIR = ML_MODELS_DIR / "vehicles"

# Directory for embedding-related files
EMBEDDING_GENERATOR = PROJECT_ROOT / "core" / "embeddings"

# Database and Vector Store URLs
POSTGRES_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/ads_db"
VECTOR_STORE_URL = "http://localhost:6333"

# Model paths
MODEL_PATHS = {
    "electronics": str(ELECTRONICS_MODELS_DIR / "electronics_cat_nn_classifier_new_2.keras"),
    "property": str(PROPERTY_MODELS_DIR / "property_classifier.keras"),
    "vehicles": str(VEHICLES_MODELS_DIR / "vehicles_classifier.keras")
}

# Batch processing settings
BATCH_SIZE = 100  # Number of items to process in each batch
MODEL_VERSION = "1.0.0"  # Current version of the classification model

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO"
        }
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO"
    }
}

