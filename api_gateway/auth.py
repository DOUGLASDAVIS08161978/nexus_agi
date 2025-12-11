"""
Authentication Module for Nexus AGI API
JWT-based authentication with user management
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_secret_key

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# JWT settings
SECRET_KEY = get_secret_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    # Bcrypt has a maximum password length of 72 bytes
    password_bytes = password.encode('utf-8')[:72]
    return pwd_context.hash(password_bytes.decode('utf-8'))


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token

    Args:
        data: Payload data to encode
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and verify JWT token

    Args:
        token: JWT token string

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Get current authenticated user from JWT token

    Args:
        credentials: HTTP authorization credentials

    Returns:
        User data from token

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    payload = decode_access_token(token)

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # TODO: Fetch user from database
    # For now, return payload data
    return {
        "user_id": user_id,
        "email": payload.get("email"),
        "tier": payload.get("tier", "free"),
        "subscription_id": payload.get("subscription_id")
    }


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, None otherwise
    Used for endpoints that work with or without authentication
    """
    if credentials is None:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def create_api_key(user_id: str, tier: str = "free") -> str:
    """
    Create an API key for a user
    API keys are long-lived tokens (1 year expiration)

    Args:
        user_id: User identifier
        tier: Subscription tier

    Returns:
        API key (JWT token)
    """
    expires_delta = timedelta(days=365)

    return create_access_token(
        data={
            "sub": user_id,
            "tier": tier,
            "type": "api_key"
        },
        expires_delta=expires_delta
    )


if __name__ == "__main__":
    # Test authentication functions
    print("=== Testing Authentication Module ===\n")

    # Test password hashing
    password = "my_secure_password123"
    hashed = get_password_hash(password)
    print(f"Password hash: {hashed[:50]}...")
    print(f"Password verification: {verify_password(password, hashed)}")
    print(f"Wrong password: {verify_password('wrong', hashed)}\n")

    # Test token creation
    token = create_access_token(
        data={"sub": "user123", "email": "user@example.com", "tier": "professional"}
    )
    print(f"JWT Token: {token[:50]}...\n")

    # Test token decoding
    payload = decode_access_token(token)
    print(f"Decoded payload: {payload}\n")

    # Test API key creation
    api_key = create_api_key("user123", "enterprise")
    print(f"API Key: {api_key[:50]}...")

    print("\n✅ Authentication module working correctly!")
