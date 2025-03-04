from fastapi import FastAPI
from app.routers import ads
import logging
from app.logging_config import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


app = FastAPI()


app.include_router(ads.router, prefix="/ads", tags=["ads"])