from typing import List, Dict, Any
from pydantic import BaseModel
from enum import Enum
from bson import ObjectId

class MainCategory(str, Enum):
    ELECTRONICS = 'electronics'
    PROPERTY = 'property'
    VEHICLE = 'vehicle'
    UNDEFINED = 'undefined'


class SubCategory(str, Enum):
    VAN = 'van'
    THREE_WHEELER = 'three-wheeler'
    ROOM_ANNEX = 'room & annex'
    LORRY_TRUCK = 'lorry_truck'
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
    OTHER_ELECTRONICS = 'other electronics'
    UNDEFINED = 'undefined'


class TransactionType(str, Enum):
    SALE = "sale"
    RENT = "rent"

class WantedOffering(str, Enum):
    OFFERING = "offering"
    WANTED = "wanted"

class AdCreate(BaseModel):
    title: str | None = None
    body: str
    user_id: str

class Ad(BaseModel):
    """Schema for an ad"""
    text: str
    main_category : MainCategory
    sub_category : SubCategory
    transaction_type : TransactionType = TransactionType.SALE
    wanted_offering : WantedOffering = WantedOffering.OFFERING
    user_id: str | None = None


class AdsRequest(BaseModel):
    """Request schema for processing multiple ads"""
    ads: List[Ad]

class MatchingAdResponse(BaseModel):
    ''' Retreived matching'''
    id : str
    text : str
    main_category : MainCategory
    sub_category : SubCategory
    transaction_type : TransactionType
    wanted_offering : WantedOffering
    score : float

class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserInDB(BaseModel):
    id: str = None
    username: str
    email: str
    hashed_password: str

class AdvertisementDto(BaseModel):
    id: str
    title: str | None = None
    url: str | None = None
    description: str | None = None
    main_category: str
    sub_category: str
    created_at: str | None = None
    transaction_type: str
    wanted_offering: str
    image_urls: list[str] = []










