from fastapi import FastAPI
import logging
from app.config.logging_config import setup_logging
from contextlib import asynccontextmanager
from app.api.endpoints.ads import router as ads_router
from app.api.endpoints.auth import router as auth_router
from fastapi.middleware.cors import CORSMiddleware


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

# Enable CORS for all origins (for development/demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ads_router, prefix="/ads", tags=["ads"])
app.include_router(auth_router, prefix="/auth", tags=["auth"])