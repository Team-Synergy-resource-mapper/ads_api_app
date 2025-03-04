import logging
import sys
from logging.handlers import RotatingFileHandler
import os


def setup_logging():
    """
    Configure logging with console and file handlers.
    
    This setup provides:
    - Logging to console (stdout)
    - Logging to a rotating file
    - Configurable log level
    - Detailed log format
    """
    # Ensure logs directory exists
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # File Handler (Rotating)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler]
    )
    
