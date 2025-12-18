"""
Configuration Loader for Nexus AGI
Securely loads environment variables and provides configuration management
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Secure configuration loader that reads from environment variables
    and provides validation and defaults
    """

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration loader

        Args:
            env_file: Path to .env file (defaults to .env in project root)
        """
        # Load environment variables
        if env_file is None:
            # Try to find .env in project root
            project_root = Path(__file__).parent.parent
            env_file = project_root / ".env"

        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded configuration from {env_file}")
        else:
            logger.warning(f".env file not found at {env_file}, using environment variables only")

        # Validate required settings
        self._validate_config()

    def _validate_config(self):
        """Validate that required configuration is present"""
        required_vars = ['SECRET_KEY']

        missing = []
        for var in required_vars:
            if not os.getenv(var) or os.getenv(var) == 'your-secret-key-here-change-in-production':
                missing.append(var)

        if missing and self.get('ENVIRONMENT') == 'production':
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
        elif missing:
            logger.warning(f"Missing configuration (acceptable in dev): {', '.join(missing)}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return os.getenv(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value"""
        value = os.getenv(key, str(default))
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer value for {key}: {value}, using default {default}")
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value"""
        value = os.getenv(key, str(default))
        try:
            return float(value)
        except ValueError:
            logger.warning(f"Invalid float value for {key}: {value}, using default {default}")
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')

    def get_list(self, key: str, separator: str = ',', default: list = None) -> list:
        """Get list configuration value (comma-separated by default)"""
        if default is None:
            default = []
        value = os.getenv(key)
        if not value:
            return default
        return [item.strip() for item in value.split(separator) if item.strip()]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.get('ENVIRONMENT', 'development').lower() == 'production'

    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.get('ENVIRONMENT', 'development').lower() == 'development'

    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Export configuration as dictionary

        Args:
            include_secrets: If False, masks sensitive values
        """
        config = {
            'app_name': self.get('APP_NAME', 'Nexus AGI'),
            'version': self.get('APP_VERSION', '4.0.0'),
            'environment': self.get('ENVIRONMENT', 'development'),
            'debug': self.get_bool('DEBUG', False),
            'api_host': self.get('API_HOST', '0.0.0.0'),
            'api_port': self.get_int('API_PORT', 8000),
            'features': {
                'payment_processing': self.get_bool('ENABLE_PAYMENT_PROCESSING', True),
                'usage_tracking': self.get_bool('ENABLE_USAGE_TRACKING', True),
                'rate_limiting': self.get_bool('ENABLE_RATE_LIMITING', True),
                'analytics': self.get_bool('ENABLE_ANALYTICS', True),
            },
            'rate_limits': {
                'free': self.get_int('RATE_LIMIT_FREE', 10),
                'starter': self.get_int('RATE_LIMIT_STARTER', 100),
                'professional': self.get_int('RATE_LIMIT_PROFESSIONAL', 1000),
                'enterprise': self.get_int('RATE_LIMIT_ENTERPRISE', 10000),
            },
            'pricing': {
                'starter': self.get_int('PRICE_STARTER', 29),
                'professional': self.get_int('PRICE_PROFESSIONAL', 99),
                'enterprise': self.get_int('PRICE_ENTERPRISE', 499),
            }
        }

        if include_secrets:
            config['secrets'] = {
                'secret_key': self.get('SECRET_KEY'),
                'stripe_secret_key': self.get('STRIPE_SECRET_KEY'),
                'database_url': self.get('DATABASE_URL'),
            }
        else:
            config['secrets'] = {
                'secret_key': '***' if self.get('SECRET_KEY') else None,
                'stripe_secret_key': '***' if self.get('STRIPE_SECRET_KEY') else None,
                'database_url': '***' if self.get('DATABASE_URL') else None,
            }

        return config


# Global configuration instance
config = ConfigLoader()


# Convenience functions for common configuration access
def get_secret_key() -> str:
    """Get JWT secret key"""
    key = config.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    if key == 'your-secret-key-here-change-in-production' and config.is_production:
        raise ValueError("SECRET_KEY must be changed in production!")
    return key


def get_database_url() -> str:
    """Get database connection URL"""
    return config.get('DATABASE_URL', 'sqlite:///./nexus_agi.db')


def get_redis_url() -> str:
    """Get Redis connection URL"""
    return config.get('REDIS_URL', 'redis://localhost:6379/0')


def get_stripe_keys() -> Dict[str, str]:
    """Get Stripe API keys"""
    return {
        'secret_key': config.get('STRIPE_SECRET_KEY'),
        'publishable_key': config.get('STRIPE_PUBLISHABLE_KEY'),
        'webhook_secret': config.get('STRIPE_WEBHOOK_SECRET'),
    }


def get_cors_origins() -> list:
    """Get allowed CORS origins"""
    return config.get_list('CORS_ORIGINS', default=['http://localhost:3000'])


def get_rate_limit(tier: str) -> int:
    """Get rate limit for subscription tier"""
    limits = {
        'free': config.get_int('RATE_LIMIT_FREE', 10),
        'starter': config.get_int('RATE_LIMIT_STARTER', 100),
        'professional': config.get_int('RATE_LIMIT_PROFESSIONAL', 1000),
        'enterprise': config.get_int('RATE_LIMIT_ENTERPRISE', 10000),
    }
    return limits.get(tier.lower(), limits['free'])


if __name__ == "__main__":
    # Test configuration loading
    print("=== Nexus AGI Configuration ===")
    print(f"Environment: {config.get('ENVIRONMENT', 'development')}")
    print(f"Debug Mode: {config.get_bool('DEBUG', False)}")
    print(f"API Port: {config.get_int('API_PORT', 8000)}")
    print(f"Features: {config.to_dict()['features']}")
    print(f"Rate Limits: {config.to_dict()['rate_limits']}")
    print(f"Pricing: {config.to_dict()['pricing']}")
    print("\nConfiguration loaded successfully!")
