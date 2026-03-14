"""
Configuration management
"""

import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Settings:
    """
    Application settings.
    
    Usage:
        settings = Settings(
            model="gpt-4",
            threshold=0.7,
            cloud_enabled=True
        )
    """
    
    # Model settings
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # Evaluation settings
    threshold: float = 0.5
    async_mode: bool = False
    
    # Cloud settings
    cloud_enabled: bool = False
    confident_api_key: Optional[str] = None
    
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Report settings
    report_format: str = "json"  # json, html, slack
    report_path: str = "./results"
    
    def __post_init__(self):
        """Load API keys from environment."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.confident_api_key = os.getenv("DEEPEVAL_API_KEY")
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls(
            model=os.getenv("AI_TEST_MODEL", "gpt-4"),
            temperature=float(os.getenv("AI_TEST_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("AI_TEST_MAX_TOKENS", "2000")),
            threshold=float(os.getenv("AI_TEST_THRESHOLD", "0.5")),
            cloud_enabled=os.getenv("DEEPEVAL_API_KEY", None) is not None,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excluding secrets)."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "threshold": self.threshold,
            "async_mode": self.async_mode,
            "cloud_enabled": self.cloud_enabled,
            "report_format": self.report_format,
            "report_path": self.report_path,
        }


# Default settings instance
default_settings = Settings()


__all__ = ["Settings", "default_settings"]
