from pydantic import BaseModel
from typing import List


class ClassificationRequest(BaseModel):
    sentences: List[str]


class ClassificationResponse(BaseModel):
    predictions: List[int]  
