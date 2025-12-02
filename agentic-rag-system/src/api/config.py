# src/api/config.py
"""API configuration."""

from pydantic import BaseSettings
from typing import List


class APISettings(BaseSettings):
    """API-specific settings."""
    
    # CORS settings
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    cors_allow_credentials: bool = True
    
    # WebSocket settings
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    
    # Session settings
    session_timeout_minutes: int = 30
    
    class Config:
        env_prefix = "API_"


api_settings = APISettings()