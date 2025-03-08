from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from app.models.models_temp import ClassificationRequest, ClassificationResponse
from app.models.models import RawListing
from app.config.setup_models import ad_classifier
from app.config.db_config import SessionLocal, DATABASE_URL
from app.models.schemas import AdsRequest, EmbeddingRequest, EmbeddingResponse
from app.services.embedding_service import EmbeddingService
from app.dependencies import get_embedding_service
from app.db.mongo_db import get_vector_db
from app.db.vector_db import VectorDB

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

    # ad_texts = [ad.text for ad in request.ads]

    # print(f"Classifying {len(ad_texts)} ads...")
    # predictions = ad_classifier.classify(ad_texts)
    # print("Completed classifying {len(predictions)} predictions")

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
    

# @router.post("/generate",
#              #   response_model=EmbeddingResponse
#              )
# async def generate_ad_embeddings(
#     request: AdsRequest,
#     embedding_service: EmbeddingService = Depends(get_embedding_service)
# ):
#     """Generate embeddings for ads using a preloaded model."""


#     if embedding_service.embedding_model is None or embedding_service.labse_model is None:
#         raise HTTPException(status_code=503, detail="Models not initialized")

#     try:
#         result = embedding_service.generate_ad_embeddings(request.ads)
#         # return result
#         return "Ads generated"
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Error generating embeddings: {str(e)}")
