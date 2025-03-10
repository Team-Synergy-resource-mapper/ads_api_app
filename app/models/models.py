from sqlalchemy import Column, Integer, String, Text, ARRAY, JSON, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

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

class BatchProcessingTracker(Base):
    __tablename__ = "batch_processing_tracker"

    id = Column(Integer, primary_key=True, autoincrement=True)
    last_processed_id = Column(Integer, nullable=False)
    processed_time = Column(DateTime, default=func.now(), nullable=False)
