"""
Configuration management using Pydantic settings.

All configuration values are loaded from environment variables or .env file.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # IBKR API Configuration
    ibkr_base_url: str = "https://localhost:5000/v1/api"
    ibkr_timeout: int = 30

    # Server Configuration
    port: int = 8080
    host: str = "127.0.0.1"
    reload: bool = False

    # Analysis Defaults
    default_bins: int = 50
    default_bootstrap: int = 0  # 0 = direct returns, >0 = Monte Carlo samples
    default_bar_size: str = "1w"  # Historical data bar size (1d, 1w, 1m)
    default_historical_years: int = 5
    default_transaction_cost: float = 0.65  # Per contract commission

    # Session Management
    session_ttl: int = 3600  # Session timeout in seconds (1 hour)
    session_cleanup_interval: int = 300  # Cleanup every 5 minutes

    # Cache Configuration
    cache_type: Literal["memory", "redis"] = "memory"
    redis_url: Optional[str] = None

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "text"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Singleton Settings instance loaded from environment.

    Note:
        Uses lru_cache to ensure settings are loaded only once.
        To reload settings (e.g., in tests), call get_settings.cache_clear()
    """
    return Settings()
