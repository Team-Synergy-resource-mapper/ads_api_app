from fastapi import FastAPI
from app.api.routers.ads import router as ads
import logging
from app.config.logging_config import setup_logging
from app.api.endpoints import ads
from contextlib import asynccontextmanager
from app.api.endpoints.ads import router as ads_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start up tasks
    print("Application startup...")
    
    yield
    
    # Clean up tasks
    print("Application shutdown...")


setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(ads_router, prefix="/ads", tags=["ads"])