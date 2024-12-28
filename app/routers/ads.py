from fastapi import APIRouter, HTTPException
from ..models import ClassificationRequest, ClassificationResponse
from ..classification import classify_ads
router = APIRouter()

@router.get("/")
def read_ads_router():
  return {"message": "Welcome to the Advertisement API"}

@router.post("/classify")
async def classify(request: ClassificationRequest):
  try:

    # Perform classification
    predictions = classify_ads(request.sentences)
    return {
      "predictions": predictions
    }
    

  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))  