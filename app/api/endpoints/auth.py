from fastapi import APIRouter, HTTPException, status, Depends
from app.models.schemas import UserRegister, UserLogin
from app.db.user_db import UserDB
from app.config import config
import bcrypt
import jwt

router = APIRouter()

user_db = UserDB(config.MONGODB_URI, config.DB_NAME)

@router.post("/auth/register")
def register(user: UserRegister):
    if user_db.user_exists(user.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    user_db.create_user(user.username, user.email, hashed_password.decode('utf-8'))
    return {"msg": "User registered successfully"}

@router.post("/auth/login")
def login(user: UserLogin):
    db_user = user_db.get_user_by_username(user.username)
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not bcrypt.checkpw(user.password.encode('utf-8'), db_user.hashed_password.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = jwt.encode({"user_id": db_user.id, "username": db_user.username}, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)
    return {"access_token": token, "token_type": "bearer"} 