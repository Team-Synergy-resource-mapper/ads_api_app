from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def read_ads_router():
  return {"message": "Welcome to the Advertisement API"}
