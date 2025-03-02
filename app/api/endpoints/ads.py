from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from ..models import ClassificationRequest, ClassificationResponse
from ...models.models import RawListing
from ...config.setup_models import ad_classifier
from ...db_config import SessionLocal, DATABASE_URL

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