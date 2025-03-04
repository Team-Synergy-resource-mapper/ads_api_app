from fastapi import FastAPI
from .routers import ads
import logging
from app.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(ads.router, prefix="/ads", tags=["ads"]) 
@app.get("/")
def read_root():
  return {"message": "Welcome to the FastAPI API!"}



