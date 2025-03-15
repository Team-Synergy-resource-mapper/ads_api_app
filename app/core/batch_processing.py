import logging
import json
from typing import Optional
from sqlalchemy.orm import Session
from app.models.models import RawListing, BatchProcessingTracker, BatchProcessingControl
from app.models.schemas import Ad, MainCategory, SubCategory, WantedOffering, TransactionType
from app.services.embedding_service import EmbeddingService
from app.db.vector_db import VectorDB
from app.config.setup_models import ad_classifier
from app.config.db_config import SessionLocal, DATABASE_URL


def reset_stop_flag(db: Session):
    """Reset the stop flag to False before starting the batch processing."""
    try:
        stop_control = db.query(BatchProcessingControl).first()
        if stop_control:
            stop_control.stop_flag = False  # Reset stop flag
            db.commit()
            logging.info("Stop flag reset to False.")
        else:
            logging.warning("No entry found in BatchProcessingControl table.")
    except Exception as e:
        logging.error(f"Error resetting stop flag: {e}")

def should_stop(db: Session) -> bool:
    """Check if the stop flag is set to True in the database."""
    stop_flag = db.query(BatchProcessingControl).first()
    return stop_flag and stop_flag.stop_flag

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
    
    # Convert RawListing fields (excluding combined_text and fetched) into a JSON string safely
    raw_listing_dict = {
        "title": listing.title or "N/A",
        "meta_data": listing.meta_data or "{}",
        "price": listing.price or "Unknown",
        "attributes": listing.attributes or [],
        "description": listing.description or "No description provided.",
        "url": listing.url or "N/A",
        "breadcrumbs": listing.breadcrumbs or [],
        "image_urls": listing.image_urls or [],
        "additional_data": listing.additional_data or {},
    }

    try:
        json_text = json.dumps(raw_listing_dict, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error serializing RawListing data: {e}")
        json_text = json.dumps({"error": "Failed to serialize listing data"})  # Fallback JSON

    return Ad(
        text=json_text,
        main_category=main_category_enum,
        sub_category=sub_category_enum,
        transaction_type=TransactionType.SALE,  # Defaulting to SALE for now
        wanted_offering=wanted_offering
    )

def process_raw_listings_batch(listings, offset, embedding_service, vector_db):
    """Process a single batch of raw listings: classify, transform, and embed."""
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
    
    # Transform each listing with their predictions to the Ad schema for storing in MongoDB
    ad_entries = generate_ad_entries_from_predictions(listings, predictions)
    if not ad_entries:
        return 0, []
    
    # Generate embeddings and store in DB
    try:
        embeddings = embedding_service.generate_ad_embeddings(ad_entries)
        vector_db.insert_ads(ad_entries, embeddings)
        logging.info(f"Processed batch starting at offset {offset}: {len(ad_entries)} ads transformed and embedded.")
        return len(ad_entries), listings
    except Exception as e:
        logging.error(f"Error during embedding generation/insertion at offset {offset}: {e}")
        return 0, []

def generate_ad_entries_from_predictions(listings, predictions):
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

    # Reset the stop flag at the start of the batch processing
    reset_stop_flag(session)
    
    offset = 0
    total_processed = 0
    last_processed_id = None  # Track last processed primary key

    # Get the last processed ID from BatchProcessingTracker table (if exists)
    try:
        last_batch_entry = session.query(BatchProcessingTracker).order_by(BatchProcessingTracker.id.desc()).first()
        if last_batch_entry:
            last_processed_id = last_batch_entry.last_processed_id
            logging.info(f"Resuming processing from last processed ID: {last_processed_id}")
        else:
            logging.info("No previous processing found. Starting fresh.")
            last_processed_id = 0  # Start from the first listing if no previous records are found
    except Exception as e:
        logging.error(f"Error fetching last processed ID: {e}")
        last_processed_id = 0  # Start from the first listing if there's an error

    try:
        while True:
            # Check if we've hit the maximum records limit
            if max_records and total_processed >= max_records:
                logging.info(f"Reached max_records limit: {max_records}. Stopping batch processing.")
                break

            # Check if the stop flag is set
            if should_stop(session):
                logging.info("Stop flag detected, halting batch processing.")
                break
                
            # Fetch the next batch of listings
            listings = session.query(RawListing).filter(RawListing.id > last_processed_id, RawListing.fetched == False).offset(offset).limit(batch_size).all()
            
            if not listings:
                logging.info("No more listings to process.")
                break
                
            # Process this batch of listings
            processed_count, processed_listings = process_raw_listings_batch(
                listings, offset, embedding_service, vector_db
            )

            if processed_listings:
                # Update fetched status for processed listings
                listing_ids = [listing.id for listing in processed_listings]
                session.query(RawListing).filter(RawListing.id.in_(listing_ids)).update({"fetched": True}, synchronize_session=False)

                # Store last processed ID
                last_processed_id = listing_ids[-1]

                # Save the batch progress in BatchProcessingTracker
                tracker_entry = BatchProcessingTracker(last_processed_id=last_processed_id)
                session.add(tracker_entry)

            session.commit()            

            total_processed += processed_count
            offset += batch_size
    except Exception as e:
        logging.error(f"Unexpected error during batch processing at offset {offset}: {e}")
    finally:
        session.close()
        logging.info("Batch transformation processing completed.")