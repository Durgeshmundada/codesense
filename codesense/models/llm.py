"""LLM key rotation models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class KeyStatus:
    """Status of a single API key in the rotation pool."""

    key: str
    is_available: bool = True
    rate_limited_at: Optional[datetime] = None
    usage_count: int = 0
    # Permanently disabled (e.g. auth/permission denied). Never auto-recovers.
    is_invalid: bool = False


@dataclass
class RotationPool:
    """Configuration and state for the key rotation pool."""

    keys: list[KeyStatus] = field(default_factory=list)
    current_index: int = 0
    retry_count: int = 0
    backoff_seconds: float = 1.0
    max_retries: int = 5
    max_backoff: float = 60.0
    cooldown_seconds: float = 60.0
