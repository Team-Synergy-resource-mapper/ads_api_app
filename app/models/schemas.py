from sqlalchemy import (
    Column, Integer, String, Text, ARRAY, JSON, Boolean, 
    DateTime, ForeignKey, Float, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class RawListing(Base):
    __tablename__ = 'raw_listings'

    id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    meta_data = Column(Text)
    price = Column(Text)
    attributes = Column(ARRAY(Text))
    description = Column(Text)
    url = Column(Text, unique=True, nullable=False)
    breadcrumbs = Column(ARRAY(Text))
    image_urls = Column(ARRAY(Text))
    additional_data = Column(JSON)
    combined_text = Column(Text)
    fetched = Column(Boolean, default=False)

    def __repr__(self):
        return f"<RawListing(title={self.title}, url={self.url})>"


class ProcessedListing(Base):
    __tablename__ = 'processed_listings'

    id = Column(Integer, primary_key=True)
    raw_listing_id = Column(Integer, ForeignKey('raw_listings.id'), nullable=False)
    
    # Classification hierarchy
    main_category = Column(String(100), nullable=False)  # e.g., Vehicle, Electronics, Property
    sub_category = Column(String(100), nullable=False)   # e.g., Cars, Phones, Apartments
    leaf_category = Column(String(100))                  # More specific categorization if available
    
    # Classification confidence scores
    main_category_confidence = Column(Float)
    sub_category_confidence = Column(Float)
    
    # Embedding vector (using PostgreSQL's vector extension)
    embedding = Column(Vector(384))  # Adjust dimension based on your model
    
    # Processing metadata
    processed_at = Column(DateTime, server_default=func.now(), nullable=False)
    model_version = Column(String(50), nullable=False)  # Version of the classification model used
    embedding_model_version = Column(String(50), nullable=False)  # Version of the embedding model
    
    # Original text used for classification
    processed_text = Column(Text, nullable=False)  # The text that was actually used for classification
    
    # Additional metadata
    classification_metadata = Column(JSON)  # Store any additional classification info
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_processed_listings_categories', 
              main_category, sub_category, leaf_category),
        Index('idx_processed_listings_raw_id', raw_listing_id),
    )

    def __repr__(self):
        return f"<ProcessedListing(id={self.id}, main_category={self.main_category}, sub_category={self.sub_category})>"