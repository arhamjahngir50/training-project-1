# pip install fastapi uvicorn python-jose[cryptography] passlib[bcrypt]
# Run: uvicorn main:app --reload

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional

# ── Config ──────────────────────────────────────────────
SECRET_KEY = "your-secret-key-change-in-prod"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ── Fake DB ─────────────────────────────────────────────
# In real projects: use SQLAlchemy + real database
fake_users_db = {
    "alice": {
        "username": "alice",
        "hashed_password": "$2b$12$...",  # will be set below
        "role": "admin",
    },
    "bob": {
        "username": "bob",
        "hashed_password": "$2b$12$...",
        "role": "user",
    },
}

# ── Password hashing ────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# Seed the fake DB with real hashes
fake_users_db["alice"]["hashed_password"] = hash_password("admin123")
fake_users_db["bob"]["hashed_password"]   = hash_password("user123")

# ── JWT helpers ─────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ── Schemas ─────────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    role: str

# ── OAuth2 scheme ────────────────────────────────────────
# This tells FastAPI where clients send the token (Authorization: Bearer <token>)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ── Dependency: get current user from token ──────────────
def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user_data = fake_users_db.get(username)
    if user_data is None:
        raise credentials_exception

    return User(username=username, role=user_data["role"])

# ── Dependency: require admin role ───────────────────────
def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user

# ── App ──────────────────────────────────────────────────
app = FastAPI(title="Auth Demo")

# LOGIN — returns JWT token
@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(
        data={"sub": form_data.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer"}

# PUBLIC — no auth needed
@app.get("/public")
def public_route():
    return {"message": "Anyone can see this"}

# PROTECTED — any logged-in user
@app.get("/me")
def read_profile(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username, "role": current_user.role}

# ADMIN ONLY — role-based authorization
@app.get("/admin")
def admin_panel(current_user: User = Depends(require_admin)):
    return {"message": f"Welcome admin {current_user.username}!"}