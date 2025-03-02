from fastapi import FastAPI
from .api.endpoints import ads
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start up tasks
    print("Application startup...")
    
    yield
    
    # Clean up tasks
    print("Application shutdown...")

app = FastAPI(lifespan=lifespan)

app.include_router(ads.router, prefix="/ads", tags=["ads"]) 
@app.get("/")
def read_root():
  return {"message": "Welcome to the FastAPI API!"}



