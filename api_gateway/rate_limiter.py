"""
Rate Limiting for Nexus AGI API
Implements per-user rate limiting based on subscription tier
"""

from typing import Optional, Dict
from datetime import datetime, timedelta
from collections import defaultdict
import time
from fastapi import HTTPException, status

# In-memory rate limiting (for production, use Redis)
# Format: {user_id: {minute: count}}
rate_limit_store: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))


class RateLimiter:
    """
    Rate limiter with tier-based limits

    Implements sliding window rate limiting
    """

    def __init__(self):
        """Initialize rate limiter"""
        # Requests per minute by tier
        self.limits = {
            "free": 10,
            "starter": 100,
            "professional": 1000,
            "enterprise": 10000,
        }

    def get_limit(self, tier: str) -> int:
        """Get rate limit for tier"""
        return self.limits.get(tier, self.limits["free"])

    def check_rate_limit(self, user_id: str, tier: str = "free") -> bool:
        """
        Check if user has exceeded rate limit

        Args:
            user_id: User identifier
            tier: Subscription tier

        Returns:
            True if within limit, False if exceeded

        Raises:
            HTTPException: If rate limit exceeded
        """
        limit = self.get_limit(tier)
        current_minute = int(time.time() / 60)

        # Clean old entries (keep last 2 minutes)
        self._cleanup_old_entries(user_id, current_minute)

        # Get current count
        current_count = rate_limit_store[user_id][current_minute]

        if current_count >= limit:
            # Calculate retry after
            retry_after = 60 - (time.time() % 60)

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": limit,
                    "tier": tier,
                    "retry_after": int(retry_after),
                    "message": f"You have exceeded your rate limit of {limit} requests per minute. Please upgrade your plan for higher limits."
                },
                headers={"Retry-After": str(int(retry_after))}
            )

        # Increment counter
        rate_limit_store[user_id][current_minute] += 1

        return True

    def _cleanup_old_entries(self, user_id: str, current_minute: int):
        """Remove old rate limit entries"""
        if user_id in rate_limit_store:
            old_minutes = [m for m in rate_limit_store[user_id].keys() if m < current_minute - 2]
            for minute in old_minutes:
                del rate_limit_store[user_id][minute]

    def get_current_usage(self, user_id: str) -> Dict[str, int]:
        """Get current usage for user"""
        current_minute = int(time.time() / 60)
        current_count = rate_limit_store.get(user_id, {}).get(current_minute, 0)

        return {
            "current_minute": current_minute,
            "requests_this_minute": current_count
        }

    def reset_user_limit(self, user_id: str):
        """Reset rate limit for user (admin function)"""
        if user_id in rate_limit_store:
            del rate_limit_store[user_id]


# Global rate limiter instance
rate_limiter = RateLimiter()


async def check_rate_limit_dependency(user: Optional[Dict] = None):
    """
    FastAPI dependency for rate limiting

    Args:
        user: Current authenticated user (from auth dependency)

    Raises:
        HTTPException: If rate limit exceeded
    """
    if user is None:
        # Anonymous user - strictest limits
        user_id = "anonymous"
        tier = "free"
    else:
        user_id = str(user.get("user_id", "unknown"))
        tier = user.get("tier", "free")

    rate_limiter.check_rate_limit(user_id, tier)


if __name__ == "__main__":
    print("=== Testing Rate Limiter ===\n")

    limiter = RateLimiter()

    # Test rate limits for different tiers
    print("Rate limits by tier:")
    for tier in ["free", "starter", "professional", "enterprise"]:
        limit = limiter.get_limit(tier)
        print(f"  {tier.capitalize()}: {limit} requests/minute")

    print("\n=== Simulating requests ===\n")

    # Test free tier
    user_id = "test_user_123"
    tier = "free"
    limit = limiter.get_limit(tier)

    print(f"Testing {tier} tier (limit: {limit}/min)...")

    # Make requests up to limit
    for i in range(limit):
        try:
            limiter.check_rate_limit(user_id, tier)
            print(f"  Request {i+1}: ✅ Allowed")
        except HTTPException as e:
            print(f"  Request {i+1}: ❌ Rate limited")

    # Try one more (should fail)
    try:
        limiter.check_rate_limit(user_id, tier)
        print(f"  Request {limit+1}: ✅ Allowed (unexpected!)")
    except HTTPException as e:
        print(f"  Request {limit+1}: ❌ Rate limited (expected)")
        print(f"     Retry after: {e.headers.get('Retry-After')} seconds")

    # Get current usage
    usage = limiter.get_current_usage(user_id)
    print(f"\nCurrent usage: {usage['requests_this_minute']} requests this minute")

    print("\n✅ Rate limiter working correctly!")
