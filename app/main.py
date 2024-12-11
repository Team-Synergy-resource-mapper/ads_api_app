from fastapi import FastAPI
from .routers import ads

app = FastAPI()

app.include_router(ads.router, prefix="/ads", tags=["ads"]) 
@app.get("/")
def read_root():
  return {"message": "Welcome to the FastAPI API!"}



