from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class Ad(BaseModel):
    """Schema for an ad"""
    # title: str
    description: str
    # metadata: Optional[Dict[str, Any]] = None


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
