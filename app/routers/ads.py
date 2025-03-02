from fastapi import APIRouter, HTTPException, Depends
# from ..models_temp import ClassificationRequest, ClassificationResponse
# from ..setup_models import ad_classifier
from app.models.schemas import AdsRequest, EmbeddingRequest, EmbeddingResponse
from app.services.embedding_service import EmbeddingService
from app.dependencies import get_embedding_service

router = APIRouter()

@router.get("/")
def read_ads_router():
  return {"message": "Welcome to the Advertisement API"}

# @router.post("/classify")
# async def classify(request: ClassificationRequest):
#   try:

#     # Perform classification
#     predictions = ad_classifier.classify(request.sentences)
#     return {
#       "predictions": predictions
#     }
    

#   except ValueError as e:
#     raise HTTPException(status_code=400, detail=str(e))  



@router.post("/generate",
            #   response_model=EmbeddingResponse
              )
async def generate_ad_embeddings(
    request: AdsRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Generate embeddings for ads using a preloaded model."""
    if embedding_service.embedding_model is None or embedding_service.labse_model is None:
        raise HTTPException(status_code=503, detail="Models not initialized")

    try:
        result = embedding_service.generate_ad_embeddings(request.ads)
        # return result
        return "Ads generated"
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating embeddings: {str(e)}")
  




