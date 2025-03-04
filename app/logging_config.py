import logging

# Configure logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # Set default level to INFO
        format="%(asctime)s - %(levelname)s - %(message)s",  # Customize format
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler("app.log")  # Log to a file
        ]
    )
