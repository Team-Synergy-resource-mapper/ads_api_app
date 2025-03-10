import logging
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from app.services.embedding_service import EmbeddingService
from app.dependencies.embedding_service import get_embedding_service
from app.dependencies.mongo_db import get_vector_db
from app.db.vector_db import VectorDB
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from app.models.models_temp import ClassificationRequest
from app.models.models import RawListing
from app.config.setup_models import ad_classifier
from app.config.db_config import SessionLocal, DATABASE_URL
from app.models.schemas import AdsRequest
from bson import ObjectId
from app.models.schemas import MatchingAdResponse
from typing import List, Optional
from app.models.schemas import MainCategory, SubCategory, WantedOffering, TransactionType, Ad

router = APIRouter()


def get_session_local():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def transform_listing_to_ad(listing: RawListing, prediction: tuple) -> Ad:
    """
    Transform a RawListing and its classification prediction into an Ad object.
    The prediction tuple is expected to be (ad_type, main_category, sub_category).
    """
    ad_type, main_category, sub_category = prediction

    # Map ad_type to WantedOffering Enum
    wanted_offering = WantedOffering.WANTED if ad_type.lower() == "wanted" else WantedOffering.OFFERING

    # Convert main_category to Enum (handling case insensitivity)
    try:
        main_category_enum = MainCategory(main_category.lower())
    except ValueError:
        logging.warning(f"Unrecognized main category: {main_category}, defaulting to ELECTRONICS.")
        main_category_enum = MainCategory.UNDEFINED  # Default fallback

    # Convert sub_category to Enum (handling case insensitivity)
    try:
        sub_category_enum = SubCategory(sub_category.lower())
    except ValueError:
        logging.warning(f"Unrecognized subcategory: {sub_category}, defaulting to ELECTRONIC_HOME_APPLIANCES.")
        sub_category_enum = SubCategory.UNDEFINED  # Default fallback

    return Ad(
        text=listing.combined_text,
        main_category=main_category_enum,
        sub_category=sub_category_enum,
        transaction_type=TransactionType.SALE,  # Defaulting to SALE for now
        wanted_offering=wanted_offering
    )


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

def process_listings_batch(listings, offset, embedding_service, vector_db):
    """Process a single batch of listings: classify, transform, and embed."""
    # Extract texts for classification from listings that have combined_text.
    texts = [listing.combined_text for listing in listings if listing.combined_text]
    if not texts:
        return 0, []
    
    try:
        # Classify the texts
        predictions = ad_classifier.classify(texts)
    except Exception as e:
        logging.error(f"Error during classification at offset {offset}: {e}")
        return 0, []
    
    # Transform each listing to the Ad schema
    ad_entries = create_ad_entries(listings, predictions)
    if not ad_entries:
        return 0, []
    
    # Generate embeddings and store in DB
    try:
        embeddings = embedding_service.generate_ad_embeddings(ad_entries)
        vector_db.insert_ads(ad_entries, embeddings)
        logging.info(f"Processed batch starting at offset {offset}: {len(ad_entries)} ads transformed and embedded.")
        return len(ad_entries), ad_entries
    except Exception as e:
        logging.error(f"Error during embedding generation/insertion at offset {offset}: {e}")
        return 0, []

def create_ad_entries(listings, predictions):
    """Create Ad objects from listings and their predictions."""
    ad_entries = []
    prediction_index = 0
    
    for listing in listings:
        if listing.combined_text:
            prediction = predictions[prediction_index]
            prediction_index += 1
            ad_entry = transform_listing_to_ad(listing, prediction)
            ad_entries.append(ad_entry)
    
    return ad_entries

def process_transform_batches(batch_size: int, max_records: Optional[int], embedding_service: EmbeddingService, vector_db: VectorDB):
    """
    Background task that processes raw listings in batches:
    1. Fetch a batch of listings.
    2. Use ad_classifier.classify on the combined_text to get predictions.
    3. Transform each listing to the Ad format.
    4. Generate embeddings and insert into MongoDB.
    5. If `max_records` is set, stops after processing that number of records.
    """
    session = SessionLocal()
    offset = 0
    total_processed = 0

    try:
        while True:
            # Check if we've hit the maximum records limit
            if max_records and total_processed >= max_records:
                logging.info(f"Reached max_records limit: {max_records}. Stopping batch processing.")
                break
                
            # Fetch the next batch of listings
            listings = session.query(RawListing).offset(offset).limit(batch_size).all()
            if not listings:
                logging.info("No more listings to process.")
                break
                
            # Process this batch of listings
            processed_count, _ = process_listings_batch(
                listings, offset, embedding_service, vector_db
            )
            
            total_processed += processed_count
            offset += batch_size
    except Exception as e:
        logging.error(f"Unexpected error during batch processing at offset {offset}: {e}")
    finally:
        session.close()
        logging.info("Batch transformation processing completed.")