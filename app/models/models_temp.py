from pydantic import BaseModel
from typing import List


class ClassificationRequest(BaseModel):
    sentences: List[str]


class ClassificationResponse(BaseModel):
    predictions: List[int]  

class Ad(BaseModel):
    description : str

class AdRequest(BaseModel):
    ads : List[Ad]   


class AdPostRequest(BaseModel):
    title: str
    description: str
    user_id: str
# class AdPostResponse(BaseModel):   
       
