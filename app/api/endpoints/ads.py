import logging
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from app.services.embedding_service import EmbeddingService
from app.dependencies.embedding_service import get_embedding_service
from app.dependencies.mongo_db import get_vector_db
from app.db.vector_db import VectorDB
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from app.models.models_temp import ClassificationRequest
from app.models.models import RawListing, BatchProcessingControl
from app.config.setup_models import ad_classifier
from app.config.db_config import SessionLocal, DATABASE_URL
from app.models.schemas import AdsRequest
from bson import ObjectId
from app.models.schemas import MatchingAdResponse
from typing import List, Optional
from app.models.schemas import MainCategory, SubCategory, WantedOffering, TransactionType, Ad
from app.core.batch_processing import transform_listing_to_ad, process_transform_batches

router = APIRouter()


def get_session_local():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/")
def read_ads_router():
  return {"message": "Welcome to the Advertisement API"}

@router.post("/classify")
async def classify(request: ClassificationRequest):
  try:

    # Perform classification
    predictions = ad_classifier.classify(request.sentences)
    return {
      "predictions": predictions
    }
    

  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))

@router.get("/raw-listings")
async def get_raw_listings(limit: int = Query(5, ge=0)):
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            query = text("SELECT * FROM raw_listings LIMIT :limit")
            result = connection.execute(query, {"limit": limit})

            listings = [dict(zip(row._mapping.keys(), row._mapping.values())) for row in result]
            return {"listings": listings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  
    

@router.get("/classify-listings")
async def classify_listings(limit: int = Query(5, ge=0), db: Session = Depends(get_session_local)):
    try:
        # Fetch raw listings from the database using the session
        listings = db.query(RawListing).limit(limit).all()

        # Extract the combined_text from each listing
        sentences = [listing.combined_text for listing in listings if listing.combined_text]

        # Perform classification
        predictions = ad_classifier.classify(sentences)

        # Combine the listings with their corresponding predictions
        classified_listings = [
            {**listing.__dict__, "prediction": prediction}
            for listing, prediction in zip(listings, predictions)
        ]

        return {
            "classified_listings": classified_listings
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate",
            #   response_model=EmbeddingResponse
              )
async def generate_ad_embeddings(
    request: AdsRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_db : VectorDB  = Depends(get_vector_db)
):
    """Generate embeddings for ads using a preloaded model."""

    if embedding_service.siamese_model is None or embedding_service.labse_model is None:
        raise HTTPException(status_code=503, detail="Models not initialized")

    try:
        result = embedding_service.generate_ad_embeddings(request.ads)
        print(result)
        print(result.shape)

        res = vector_db.insert_ads(request.ads, result)
        print(res)
        # return result
        return f"{len(result)} ads generated."
    
        
    except Exception as e:
        print("Error occurred while inserting ads into vector database:")
        print(e)
        raise HTTPException(
            status_code=500, detail=f"Error generating embeddings: {str(e)}")
    

@router.get("/ads/similar/{ad_id}", response_model=List[MatchingAdResponse])
def get_similar_ads(ad_id: str, 
                    limit: int = 10, 
                    num_candidates: int = 100, 
                    vector_db :VectorDB = Depends(get_vector_db)):
    try:
        object_id = ObjectId(ad_id)  # Validate ObjectId
    except:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    similar_ads = vector_db.search_similar_by_ad_id(
        object_id, limit, num_candidates)

    if not similar_ads:
        raise HTTPException(status_code=404, detail="No similar ads found")

    return [
        MatchingAdResponse(
            id=str(ad["_id"]),
            text=ad.get("text", ""),
            main_category=ad.get("main_category", ""),
            sub_category=ad.get("sub_category", ""),
            transaction_type=ad.get("transaction_type", ""),
            wanted_offering=ad.get("wanted_offering", ""),
            score=ad["score"]
        ) for ad in similar_ads
    ]

@router.post("/batch-transform-embeddings")
async def batch_transform_embeddings(
    background_tasks: BackgroundTasks,
    batch_size: int = Query(100, ge=1, description="Number of records to process in each batch"),
    max_records: Optional[int] = Query(None, ge=1, description="Maximum number of records to process (optional)"),
    db: Session = Depends(get_session_local),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_db: VectorDB = Depends(get_vector_db)
):
    """
    Endpoint to process raw listings in batches:
    - Classify listings using combined_text with ad_classifier.
    - Transform them to the Ad schema.
    - Generate embeddings and store the results in MongoDB.
    - Can optionally stop after processing `max_records`.
    """
    if not embedding_service.siamese_model or not embedding_service.labse_model:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    # Launch the batch processing as a background task.
    background_tasks.add_task(process_transform_batches, batch_size, max_records, embedding_service, vector_db)
    return {"message": "Batch transformation started in the background."}

@router.post("/stop_batch_processing")
async def stop_batch_processing(
    stop: bool = Query(True, description="Set to True to stop the batch"),
    db: Session = Depends(get_session_local)
):
    try:
        # Set the stop flag to True to stop the batch processing
        stop_flag = db.query(BatchProcessingControl).first()
        if not stop_flag:
            stop_flag = BatchProcessingControl(stop_flag=True)
            db.add(stop_flag)
        else:
            stop_flag.stop_flag = True
        db.commit()
        return {"message": "Batch processing has been stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping batch processing: {e}")