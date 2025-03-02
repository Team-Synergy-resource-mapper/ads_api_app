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

