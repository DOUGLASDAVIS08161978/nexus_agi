"""
Database Models for Nexus AGI API
SQLAlchemy models for users, subscriptions, and usage tracking
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_database_url

# Database setup
DATABASE_URL = get_database_url()

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base model
Base = declarative_base()


class User(Base):
    """User model"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    subscriptions = relationship("Subscription", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user")
    usage_records = relationship("UsageRecord", back_populates="user")


class Subscription(Base):
    """Subscription model"""
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    stripe_customer_id = Column(String, index=True)
    stripe_subscription_id = Column(String, unique=True, index=True)
    stripe_price_id = Column(String)

    tier = Column(String, default="free")  # free, starter, professional, enterprise
    status = Column(String, default="active")  # active, canceled, past_due, trialing
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)

    requests_limit = Column(Integer, default=100)  # Requests per month
    requests_used = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    canceled_at = Column(DateTime)

    # Relationships
    user = relationship("User", back_populates="subscriptions")


class APIKey(Base):
    """API Key model"""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String)  # User-friendly name for the key
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime)
    expires_at = Column(DateTime)

    # Relationships
    user = relationship("User", back_populates="api_keys")


class UsageRecord(Base):
    """Usage tracking model"""
    __tablename__ = "usage_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    tool_name = Column(String, index=True)
    endpoint = Column(String)

    request_params = Column(JSON)  # Store request parameters
    response_status = Column(String)  # success, error
    execution_time = Column(Float)  # Execution time in seconds

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User", back_populates="usage_records")


class Payment(Base):
    """Payment tracking model"""
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    stripe_payment_id = Column(String, unique=True, index=True)
    stripe_invoice_id = Column(String)

    amount = Column(Float)  # Amount in USD
    currency = Column(String, default="usd")
    status = Column(String)  # succeeded, failed, pending
    description = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")


def get_user_by_email(db, email: str):
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db, user_id: int):
    """Get user by ID"""
    return db.query(User).filter(User.id == user_id).first()


def create_user(db, email: str, hashed_password: str, full_name: str = None):
    """Create new user"""
    user = User(
        email=email,
        hashed_password=hashed_password,
        full_name=full_name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_subscription(db, user_id: int, tier: str = "free"):
    """Create subscription for user"""
    # Set limits based on tier
    limits = {
        "free": 100,
        "starter": 1000,
        "professional": 10000,
        "enterprise": 100000
    }

    subscription = Subscription(
        user_id=user_id,
        tier=tier,
        requests_limit=limits.get(tier, 100),
        status="active"
    )
    db.add(subscription)
    db.commit()
    db.refresh(subscription)
    return subscription


def get_active_subscription(db, user_id: int):
    """Get user's active subscription"""
    return db.query(Subscription).filter(
        Subscription.user_id == user_id,
        Subscription.status == "active"
    ).first()


def log_usage(db, user_id: int, tool_name: str, endpoint: str, params: dict, status: str, execution_time: float):
    """Log API usage"""
    record = UsageRecord(
        user_id=user_id,
        tool_name=tool_name,
        endpoint=endpoint,
        request_params=params,
        response_status=status,
        execution_time=execution_time
    )
    db.add(record)
    db.commit()

    # Increment usage counter
    subscription = get_active_subscription(db, user_id)
    if subscription:
        subscription.requests_used += 1
        db.commit()


def get_usage_stats(db, user_id: int):
    """Get usage statistics for user"""
    subscription = get_active_subscription(db, user_id)

    if not subscription:
        return {
            "tier": "free",
            "requests_used": 0,
            "requests_limit": 100,
            "requests_remaining": 100,
            "usage_percentage": 0
        }

    remaining = subscription.requests_limit - subscription.requests_used
    percentage = (subscription.requests_used / subscription.requests_limit) * 100 if subscription.requests_limit > 0 else 0

    return {
        "tier": subscription.tier,
        "requests_used": subscription.requests_used,
        "requests_limit": subscription.requests_limit,
        "requests_remaining": max(0, remaining),
        "usage_percentage": round(percentage, 2),
        "period_start": subscription.current_period_start,
        "period_end": subscription.current_period_end
    }


if __name__ == "__main__":
    print("=== Initializing Nexus AGI Database ===\n")

    # Initialize database
    init_db()

    # Test database operations
    db = SessionLocal()

    try:
        # Create test user
        print("Creating test user...")
        from api_gateway.auth import get_password_hash

        user = create_user(
            db,
            email="test@nexusagi.com",
            hashed_password=get_password_hash("testpassword123"),
            full_name="Test User"
        )
        print(f"✅ User created: {user.email} (ID: {user.id})\n")

        # Create subscription
        print("Creating subscription...")
        subscription = create_subscription(db, user.id, tier="professional")
        print(f"✅ Subscription created: {subscription.tier} tier")
        print(f"   Limit: {subscription.requests_limit} requests/month\n")

        # Log some usage
        print("Logging usage...")
        log_usage(db, user.id, "web_search", "/api/v1/execute/web_search", {"query": "test"}, "success", 0.5)
        log_usage(db, user.id, "http_fetch", "/api/v1/execute/http_fetch", {"url": "https://example.com"}, "success", 0.3)
        print("✅ Usage logged\n")

        # Get usage stats
        print("Usage statistics:")
        stats = get_usage_stats(db, user.id)
        print(f"   Tier: {stats['tier']}")
        print(f"   Used: {stats['requests_used']}/{stats['requests_limit']}")
        print(f"   Remaining: {stats['requests_remaining']}")
        print(f"   Usage: {stats['usage_percentage']}%\n")

        print("✅ Database operations successful!")

    finally:
        db.close()
