"""Key rotation manager for multi-key Gemini API access.

Implements round-robin key selection with independent rate-limit tracking
and automatic cooldown recovery, plus exponential backoff retry when all
keys are rate-limited.
"""

import os
import time
from datetime import datetime, timezone

from codesense.models.llm import KeyStatus, RotationPool


class KeyRotator:
    """Manages a pool of API keys with round-robin selection and rate-limit tracking.

    Keys are read from environment variables GEMINI_KEY_1, GEMINI_KEY_2, ..., GEMINI_KEY_N.
    When a key is rate-limited, it enters a 60-second cooldown before becoming available again.
    """

    def __init__(self, api_keys: list[str] | None = None) -> None:
        """Initialize the KeyRotator.

        Args:
            api_keys: Optional list of API keys. If not provided, keys are read
                from environment variables GEMINI_KEY_1, GEMINI_KEY_2, ..., GEMINI_KEY_N.

        Raises:
            ValueError: If no API keys are configured.
        """
        if api_keys is None:
            api_keys = self._load_keys_from_env()

        if not api_keys:
            raise ValueError(
                "No Gemini API keys configured. "
                "Set at least one environment variable: GEMINI_KEY_1, GEMINI_KEY_2, ..., GEMINI_KEY_N"
            )

        key_statuses = [KeyStatus(key=key) for key in api_keys]
        self._pool = RotationPool(keys=key_statuses)

    @staticmethod
    def _load_keys_from_env() -> list[str]:
        """Load API keys from GEMINI_KEY_1, GEMINI_KEY_2, ..., GEMINI_KEY_N environment variables."""
        keys: list[str] = []
        index = 1
        while True:
            key = os.environ.get(f"GEMINI_KEY_{index}")
            if key is None:
                break
            if key.strip():
                keys.append(key.strip())
            index += 1
        return keys

    def get_next_key(self) -> str:
        """Get the next available API key using strict round-robin selection.

        Iterates through the pool starting from the current index. Keys that have
        been rate-limited for >= 60 seconds are automatically recovered. Skips keys
        that are still within their cooldown period.

        Returns:
            The next available API key string.

        Raises:
            ValueError: If no API keys are configured (should not happen after __init__).
        """
        pool = self._pool
        n = len(pool.keys)

        # Try each key starting from current_index in round-robin order
        for _ in range(n):
            key_status = pool.keys[pool.current_index]

            # Auto-recover keys past their cooldown (invalid keys never recover)
            if (
                not key_status.is_available
                and not key_status.is_invalid
                and key_status.rate_limited_at is not None
            ):
                elapsed = (
                    datetime.now(timezone.utc) - key_status.rate_limited_at
                ).total_seconds()
                if elapsed >= pool.cooldown_seconds:
                    key_status.is_available = True
                    key_status.rate_limited_at = None

            if key_status.is_available:
                key_status.usage_count += 1
                selected_key = key_status.key
                # Advance index for next call (strict round-robin)
                pool.current_index = (pool.current_index + 1) % n
                return selected_key

            # Key not available, try next
            pool.current_index = (pool.current_index + 1) % n

        # All keys are rate-limited (none recovered)
        raise RuntimeError(
            "All API keys are currently rate-limited. "
            "Please wait for cooldown period to expire or add more keys."
        )

    def get_key_with_retry(self) -> str:
        """Get the next available API key, retrying with exponential backoff if all are rate-limited.

        When all keys are rate-limited, applies exponential backoff starting at 1 second,
        doubling on each attempt (1s, 2s, 4s, 8s, 16s...), capped at 60 seconds maximum.
        Retries up to 5 times. After each backoff wait, rechecks if any key has recovered
        (past 60s cooldown).

        Returns:
            The next available API key string.

        Raises:
            RuntimeError: If all API keys remain rate-limited after exhausting all 5 retry attempts.
        """
        max_retries = self._pool.max_retries
        max_backoff = self._pool.max_backoff
        backoff = 1.0

        # Total attempts must equal max_retries. The final attempt happens
        # after the last backoff wait, so we loop (max_retries - 1) times
        # before it.
        for _ in range(max_retries - 1):
            try:
                return self.get_next_key()
            except RuntimeError:
                # All keys are rate-limited; apply exponential backoff
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

        # Final attempt after all retries with backoff
        try:
            return self.get_next_key()
        except RuntimeError:
            raise RuntimeError(
                "All API keys are rate-limited and maximum retry attempts (5) have been exhausted. "
                "Please wait for cooldown period to expire or add more keys."
            )

    def mark_rate_limited(self, key: str) -> None:
        """Mark a specific key as rate-limited, recording the current timestamp.

        Args:
            key: The API key string to mark as rate-limited.
        """
        for key_status in self._pool.keys:
            if key_status.key == key:
                key_status.is_available = False
                key_status.rate_limited_at = datetime.now(timezone.utc)
                return

    def mark_available(self, key: str) -> None:
        """Mark a specific key as available, clearing its rate-limit status.

        Args:
            key: The API key string to mark as available.
        """
        for key_status in self._pool.keys:
            if key_status.key == key:
                key_status.is_available = True
                key_status.rate_limited_at = None
                return

    def mark_invalid(self, key: str) -> None:
        """Permanently disable a key (e.g. auth/permission denied).

        Unlike rate-limiting, an invalid key never auto-recovers, so the
        rotator will skip it for the remainder of the process. This prevents
        a single dead key from failing a large fraction of calls in a pool.

        Args:
            key: The API key string to disable.
        """
        for key_status in self._pool.keys:
            if key_status.key == key:
                key_status.is_available = False
                key_status.is_invalid = True
                key_status.rate_limited_at = None
                return

    def has_usable_keys(self) -> bool:
        """Return True if at least one key is not permanently invalid."""
        return any(not ks.is_invalid for ks in self._pool.keys)

    def is_all_rate_limited(self) -> bool:
        """Check if all keys in the pool are currently rate-limited.

        This method also performs cooldown recovery: keys that have been
        rate-limited for >= 60 seconds are automatically marked as available.

        Returns:
            True if all keys are rate-limited (after recovery check), False otherwise.
        """
        pool = self._pool
        now = datetime.now(timezone.utc)

        for key_status in pool.keys:
            if key_status.is_available:
                return False
            # Check cooldown recovery
            if key_status.rate_limited_at is not None:
                elapsed = (now - key_status.rate_limited_at).total_seconds()
                if elapsed >= pool.cooldown_seconds:
                    key_status.is_available = True
                    key_status.rate_limited_at = None
                    return False

        return True

    @property
    def pool(self) -> RotationPool:
        """Access the underlying rotation pool state."""
        return self._pool

    @property
    def available_keys(self) -> list[str]:
        """Get the list of currently available (non-rate-limited) keys."""
        now = datetime.now(timezone.utc)
        available = []
        for key_status in self._pool.keys:
            if key_status.is_available:
                available.append(key_status.key)
            elif key_status.rate_limited_at is not None:
                elapsed = (now - key_status.rate_limited_at).total_seconds()
                if elapsed >= self._pool.cooldown_seconds:
                    available.append(key_status.key)
        return available
