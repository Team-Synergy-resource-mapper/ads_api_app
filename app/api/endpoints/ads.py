import logging
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, Header
from app.services.embedding_service import EmbeddingService
from app.dependencies.embedding_service import get_embedding_service
from app.dependencies.mongo_db import get_vector_db
from app.db.vector_db import VectorDB
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from app.models.models_temp import ClassificationRequest, AdPostRequest
from app.models.models import RawListing, BatchProcessingControl
from app.config.setup_models import ad_classifier
from app.config.db_config import SessionLocal, DATABASE_URL
from app.models.schemas import AdsRequest, AdCreate, AdvertisementDto
from bson import ObjectId
from app.models.schemas import MatchingAdResponse
from typing import List, Optional
from app.models.schemas import MainCategory, SubCategory, WantedOffering, TransactionType, Ad
from app.core.batch_processing import transform_listing_to_ad, process_transform_batches
import jwt
from app.config import config
import json

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


@router.post("/post_deprecated")
async def classify(request: AdPostRequest,
                   embedding_service: EmbeddingService = Depends(
                       get_embedding_service),
                   vector_db: VectorDB = Depends(get_vector_db)):
  try:

    # Perform classification
    if not request.title or not request.description:
        raise ValueError("Title and description are required fields.")
    if not request.user_id:
        raise ValueError("User ID is required.")
    
    ad_text = f"{request.title} {request.description}"

    predictions = ad_classifier.classify([ad_text])
    if not predictions or len(predictions) == 0:
        raise ValueError("No predictions were made. Please check the input data.")
    print(predictions)
    # Extract predictions
    wanted_offering, main_category, sub_category = [item.lower() for item in predictions[0]]
    ad = Ad(
        text=ad_text,
        main_category=MainCategory(main_category),
        sub_category=SubCategory(sub_category),
        transaction_type=TransactionType.SALE,  # Default to SALE
        wanted_offering=WantedOffering(wanted_offering),
        user_id=request.user_id 
    )

    # Generate embeddings for the ad
    if embedding_service.siamese_model is None or embedding_service.labse_model is None:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    embeddings = embedding_service.generate_ad_embeddings([ad])

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Failed to generate embeddings for the ad.")
    
    # Insert the ad into the vector database
    result = vector_db.insert_ads([ad], embeddings)
    if not result:
        raise ValueError("Failed to insert ad into the vector database.")
    

    return {
        "message": "Ad classified and embeddings generated successfully.",
        "ad_id": str(result.inserted_ids[0]),  # Assuming the result contains inserted_ids
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
    

def convert_to_dto(ad):
    try:
        text_json = json.loads(ad.get('text', '{}')) if isinstance(ad.get('text'), str) else ad.get('text', {})
    except Exception:
        text_json = {}
    print(text_json)
    return AdvertisementDto(
        id=str(ad.get('_id', ad.get('id', ''))),
        title=text_json.get('title'),
        url=text_json.get('url'),
        description=text_json.get('description'),
        main_category=str(ad.get('main_category', '')),
        sub_category=str(ad.get('sub_category', '')),
        created_at=ad.get('created_at').isoformat() if ad.get('created_at') else None,
        transaction_type=str(ad.get('transaction_type', '')),
        wanted_offering=str(ad.get('wanted_offering', '')),
        image_urls=text_json.get('image_urls', [])
    )

@router.get("/ads/similar/{ad_id}", response_model=list[AdvertisementDto])
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

    return [convert_to_dto(ad) for ad in similar_ads]

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

# Dependency to get current user from JWT
async def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.post("/post_ad")
async def post_ad(
    ad: AdCreate,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_db: VectorDB = Depends(get_vector_db)
):
    # Combine title and body
    combined_text = (ad.title + " ") if ad.title else ""
    combined_text += ad.body
    # Classify
    predictions = ad_classifier.classify([combined_text])
    print("Predictions:", predictions)
    if not predictions or not isinstance(predictions[0], (list, tuple)):
        raise ValueError("Classifier did not return a valid prediction tuple.")
    pred = [item.lower() for item in predictions[0]]
    if len(pred) == 4:
        main_category, sub_category, transaction_type, wanted_offering = pred
    elif len(pred) == 3:
        wanted_offering, main_category, sub_category = pred
        transaction_type = "sale"  # Default
    else:
        raise ValueError(f"Classifier returned {len(pred)} values, expected 3 or 4.")
    # Store title and description as JSON in text field
    text_json = json.dumps({
        "title": ad.title,
        "description": ad.body,
        "url": None,
        "image_urls": []
    })
    ad_obj = Ad(
        text=text_json,
        main_category=MainCategory(main_category),
        sub_category=SubCategory(sub_category),
        transaction_type=TransactionType(transaction_type),
        wanted_offering=WantedOffering(wanted_offering),
        user_id=ad.user_id
    )
    # Generate embedding
    embedding = embedding_service.generate_ad_embeddings([ad_obj])[0]
    # Store in DB
    result = vector_db.insert_ad(ad_obj, embedding)
    return {"msg": "Ad posted successfully", "ad_id": str(result.inserted_id),}

@router.get("/my", response_model=list[AdvertisementDto])
async def get_my_ads(
    user_id: str,
    vector_db: VectorDB = Depends(get_vector_db)
):
    ads = list(vector_db.collection.find({"user_id": user_id}))
    return [convert_to_dto(ad) for ad in ads]

@router.delete("/delete/{ad_id}")
async def delete_ad(
    ad_id: str,
    vector_db: VectorDB = Depends(get_vector_db)
):
    try:
        object_id = ObjectId(ad_id)  # Validate ObjectId
    except:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    result = vector_db.collection.delete_one({"_id": object_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Ad not found")
    
    return {"msg": "Ad deleted successfully"}

@router.get("/filter")
async def filter_ads(
    main_category: Optional[MainCategory] = None,
    sub_category: Optional[SubCategory] = None,
    transaction_type: Optional[TransactionType] = None,
    wanted_offering: Optional[WantedOffering] = None,
    limit: int = Query(10, ge=1),
    vector_db: VectorDB = Depends(get_vector_db)
):
    query = {}
    if main_category:
        query["main_category"] = main_category.value
    if sub_category:
        query["sub_category"] = sub_category.value
    if transaction_type:
        query["transaction_type"] = transaction_type.value
    if wanted_offering:
        query["wanted_offering"] = wanted_offering.value

    ads = list(vector_db.collection.find(query).limit(limit))
    return [convert_to_dto(ad) for ad in ads]

