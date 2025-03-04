from fastapi import FastAPI
from .routers import ads
import logging
from app.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start up tasks
    print("Application startup...")
    
    yield
    
    # Clean up tasks
    print("Application shutdown...")

app = FastAPI(lifespan=lifespan)

app.include_router(ads_router, prefix="/ads", tags=["ads"])