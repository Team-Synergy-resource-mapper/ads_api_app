from fastapi import FastAPI
from .api.endpoints import ads, batch
from .services.batch_service import BatchProcessingService
from .storage.vector_store import VectorStore
from .storage.postgres import PostgresClient
from .core.ad_classifier import AdClassifier
from .config.settings import (
    POSTGRES_URL,
    VECTOR_STORE_URL,
    MODEL_PATHS
)
from contextlib import asynccontextmanager
from .config.setup_models import ad_classifier

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize connections
    await postgres_client.initialize()
    await vector_store.initialize()
    yield
    # Clean up connections
    await postgres_client.cleanup()
    await vector_store.cleanup()

app = FastAPI(lifespan=lifespan)

# Initialize components
postgres_client = PostgresClient(POSTGRES_URL)
vector_store = VectorStore(VECTOR_STORE_URL)

# Initialize batch service
batch_service = BatchProcessingService(
    postgres_client=postgres_client,
    vector_store=vector_store,
    classifier=ad_classifier
)

# Include routers
app.include_router(ads.router, prefix="/ads", tags=["ads"])
app.include_router(batch.router, prefix="/batch", tags=["batch"])

# Make batch service available to endpoints
app.state.batch_service = batch_service

@app.get("/")
def read_root():
  return {"message": "Welcome to the FastAPI API!"}

# # Initialize the model manager and ad classifier
# manager = ModelManager(categories=["Main", "Vehicle", "Electronics", "Property"], label_to_category={})
# ad_classifier = AdClassifier(manager)

# # Initialize the batch processor but don't start it yet
# batch_processor = BatchProcessor(ad_classifier, interval_seconds=3600)  # Run every hour (adjust as needed)


# # Control Endpoints to Start/Stop Batch Processing
# @app.post("/batch/start")
# def start_batch_processing():
#     """Start the batch processing."""
#     try:
#         batch_processor.start()
#         return {"message": "Batch processor started."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

# @app.post("/batch/stop")
# def stop_batch_processing():
#     """Stop the batch processing."""
#     try:
#         batch_processor.stop()
#         return {"message": "Batch processor stopped."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
