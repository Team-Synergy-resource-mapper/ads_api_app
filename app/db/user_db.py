import logging
from pymongo import MongoClient
from app.config import config
from app.models.schemas import UserInDB
from bson import ObjectId

logger = logging.getLogger(__name__)

class UserDB:
    def __init__(self, mongodb_uri: str, db_name: str, collection_name: str = "users"):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def create_user(self, username: str, email: str, hashed_password: str) -> str:
        user = {
            "username": username,
            "email": email,
            "hashed_password": hashed_password
        }
        result = self.collection.insert_one(user)
        logger.info(f"User created with id: {result.inserted_id}")
        return str(result.inserted_id)

    def get_user_by_username(self, username: str) -> UserInDB | None:
        user = self.collection.find_one({"username": username})
        if user:
            return UserInDB(
                id=str(user["_id"]),
                username=user["username"],
                email=user["email"],
                hashed_password=user["hashed_password"]
            )
        return None

    def user_exists(self, username: str) -> bool:
        return self.collection.count_documents({"username": username}) > 0 