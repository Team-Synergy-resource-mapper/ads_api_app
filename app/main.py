from fastapi import FastAPI
from .api.endpoints import ads
from contextlib import asynccontextmanager
from .api.endpoints.ads import router as ads_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start up tasks
    print("Application startup...")
    
    yield
    
    # Clean up tasks
    print("Application shutdown...")

app = FastAPI(lifespan=lifespan)

app.include_router(ads_router, prefix="/ads", tags=["ads"])