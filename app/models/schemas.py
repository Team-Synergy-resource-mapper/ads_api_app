from typing import List, Dict, Any
from pydantic import BaseModel
from enum import Enum


class MainCategory(str, Enum):
    ELECTRONICS = 'electronics'
    PROPERTY = 'property'
    VEHICLE = 'vehicle'


class SubCategory(str, Enum):
    VAN = 'van'
    THREE_WHEELER = 'three-wheeler'
    ROOM_ANNEX = 'room & annex'
    LORRY_TRUCK = 'lorry truck'
    LAND = 'land'
    HOUSE = 'house'
    COMMERCIAL_PROPERTY = 'commercial property'
    CAR = 'car'
    BIKE = 'bike'
    BICYCLE = 'bicycle'
    APARTMENT = 'apartment'
    TVS = 'tvs'
    MOBILE_PHONES_TABLETS = 'mobile phones & tablets'
    MOBILE_PHONE_ACCESSORIES = 'mobile phone accessories'
    ELECTRONIC_HOME_APPLIANCES = 'electronic home appliances'
    COMPUTERS = 'computers'
    COMPUTER_ACCESSORIES = 'computer accessories'
    CAMERAS_CAMCORDERS = 'cameras & camcorders'
    AUDIO_MP3 = 'audio & mp3'
    AIR_CONDITIONS_ELECTRICAL_FITTINGS = 'air conditions & electrical fittings'


class TransactionType(str, Enum):
    SALE = "sale"
    RENT = "rent"


class WantedOffering(str, Enum):
    OFFERING = "offering"
    WANTED = "wanted"
class Ad(BaseModel):
    """Schema for an ad"""
    text: str
    main_category : MainCategory
    sub_category : SubCategory
    transaction_type : TransactionType = TransactionType.SALE
    wanted_offering : WantedOffering = WantedOffering.OFFERING



class AdsRequest(BaseModel):
    """Request schema for processing multiple ads"""
    ads: List[Ad]


class QueryRequest(BaseModel):
    """Request schema for search queries"""
    query: str
    top_k: int = 10


class EmbeddingRequest(BaseModel):
    """Request schema for text embedding generation"""
    text: List[str]


class SearchResult(BaseModel):
    """Schema for search results"""
    id: int
    similarity: float
    ad_info: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response schema for search results"""
    results: List[SearchResult]


class EmbeddingResponse(BaseModel):
    """Response schema for embedding generation"""
    message: str
    labse_embeddings: List[List[float]]
    final_embeddings: List[List[float]]
    ad_data: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Response schema for health check"""
    embedding_model: bool
    labse_model: bool
    vector_db: bool
    ad_count: int
