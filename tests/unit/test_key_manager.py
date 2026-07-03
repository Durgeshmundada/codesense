"""Unit tests for KeyRotator."""

import os
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from codesense.llm.key_manager import KeyRotator


class TestKeyRotatorInit:
    """Test KeyRotator initialization."""

    def test_init_with_explicit_keys(self):
        rotator = KeyRotator(api_keys=["key-1", "key-2", "key-3"])
        assert len(rotator.pool.keys) == 3
        assert rotator.pool.keys[0].key == "key-1"
        assert rotator.pool.keys[1].key == "key-2"
        assert rotator.pool.keys[2].key == "key-3"

    def test_init_with_empty_keys_raises_value_error(self):
        with pytest.raises(ValueError, match="No Gemini API keys configured"):
            KeyRotator(api_keys=[])

    def test_init_with_no_env_keys_raises_value_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No Gemini API keys configured"):
                KeyRotator(api_keys=None)

    def test_init_loads_from_env_variables(self):
        env = {"GEMINI_KEY_1": "env-key-a", "GEMINI_KEY_2": "env-key-b"}
        with patch.dict(os.environ, env, clear=True):
            rotator = KeyRotator()
            assert len(rotator.pool.keys) == 2
            assert rotator.pool.keys[0].key == "env-key-a"
            assert rotator.pool.keys[1].key == "env-key-b"

    def test_init_skips_empty_env_keys(self):
        env = {"GEMINI_KEY_1": "valid-key", "GEMINI_KEY_2": "  ", "GEMINI_KEY_3": "another-key"}
        with patch.dict(os.environ, env, clear=True):
            rotator = KeyRotator()
            assert len(rotator.pool.keys) == 2
            assert rotator.pool.keys[0].key == "valid-key"
            assert rotator.pool.keys[1].key == "another-key"

    def test_init_stops_at_first_missing_env_key(self):
        env = {"GEMINI_KEY_1": "first", "GEMINI_KEY_3": "third"}
        with patch.dict(os.environ, env, clear=True):
            rotator = KeyRotator()
            # Should only get key 1, since key 2 is missing and iteration stops
            assert len(rotator.pool.keys) == 1
            assert rotator.pool.keys[0].key == "first"


class TestRoundRobin:
    """Test strict round-robin key selection."""

    def test_strict_round_robin_order(self):
        rotator = KeyRotator(api_keys=["a", "b", "c"])
        results = [rotator.get_next_key() for _ in range(9)]
        assert results == ["a", "b", "c", "a", "b", "c", "a", "b", "c"]

    def test_single_key_always_returns_same(self):
        rotator = KeyRotator(api_keys=["only-key"])
        results = [rotator.get_next_key() for _ in range(5)]
        assert all(k == "only-key" for k in results)

    def test_two_keys_alternate(self):
        rotator = KeyRotator(api_keys=["x", "y"])
        results = [rotator.get_next_key() for _ in range(6)]
        assert results == ["x", "y", "x", "y", "x", "y"]


class TestRateLimiting:
    """Test rate-limit tracking and recovery."""

    def test_mark_rate_limited_records_timestamp(self):
        rotator = KeyRotator(api_keys=["a", "b"])
        rotator.mark_rate_limited("a")
        assert rotator.pool.keys[0].is_available is False
        assert rotator.pool.keys[0].rate_limited_at is not None

    def test_rate_limited_key_skipped_in_rotation(self):
        rotator = KeyRotator(api_keys=["a", "b", "c"])
        rotator.mark_rate_limited("b")
        keys = [rotator.get_next_key() for _ in range(4)]
        assert "b" not in keys
        assert keys == ["a", "c", "a", "c"]

    def test_mark_available_clears_status(self):
        rotator = KeyRotator(api_keys=["a", "b"])
        rotator.mark_rate_limited("a")
        rotator.mark_available("a")
        assert rotator.pool.keys[0].is_available is True
        assert rotator.pool.keys[0].rate_limited_at is None

    def test_rate_limit_independence(self):
        """Rate-limiting key X does not affect other keys."""
        rotator = KeyRotator(api_keys=["a", "b", "c"])
        rotator.mark_rate_limited("b")
        assert rotator.pool.keys[0].is_available is True
        assert rotator.pool.keys[1].is_available is False
        assert rotator.pool.keys[2].is_available is True

    def test_all_rate_limited_raises_runtime_error(self):
        rotator = KeyRotator(api_keys=["a", "b"])
        rotator.mark_rate_limited("a")
        rotator.mark_rate_limited("b")
        with pytest.raises(RuntimeError, match="All API keys are currently rate-limited"):
            rotator.get_next_key()


class TestCooldownRecovery:
    """Test 60-second cooldown auto-recovery."""

    def test_key_recovered_after_60_seconds(self):
        rotator = KeyRotator(api_keys=["a", "b"])
        rotator.mark_rate_limited("a")
        # Simulate 61 seconds elapsed
        rotator.pool.keys[0].rate_limited_at = datetime.now(timezone.utc) - timedelta(seconds=61)
        key = rotator.get_next_key()
        assert key == "a"
        assert rotator.pool.keys[0].is_available is True

    def test_key_not_recovered_before_60_seconds(self):
        rotator = KeyRotator(api_keys=["a", "b"])
        rotator.mark_rate_limited("a")
        # Simulate 30 seconds elapsed (not enough)
        rotator.pool.keys[0].rate_limited_at = datetime.now(timezone.utc) - timedelta(seconds=30)
        key = rotator.get_next_key()
        assert key == "b"  # Should skip "a"

    def test_is_all_rate_limited_with_cooldown_recovery(self):
        rotator = KeyRotator(api_keys=["a", "b"])
        rotator.mark_rate_limited("a")
        rotator.mark_rate_limited("b")
        # One key past cooldown
        rotator.pool.keys[0].rate_limited_at = datetime.now(timezone.utc) - timedelta(seconds=61)
        assert rotator.is_all_rate_limited() is False

    def test_is_all_rate_limited_true_within_cooldown(self):
        rotator = KeyRotator(api_keys=["a", "b"])
        rotator.mark_rate_limited("a")
        rotator.mark_rate_limited("b")
        # Both within cooldown
        rotator.pool.keys[0].rate_limited_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        rotator.pool.keys[1].rate_limited_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        assert rotator.is_all_rate_limited() is True


class TestGetKeyWithRetry:
    """Test exponential backoff retry logic."""

    def test_returns_key_immediately_when_available(self):
        rotator = KeyRotator(api_keys=["a", "b", "c"])
        key = rotator.get_key_with_retry()
        assert key == "a"

    @patch("codesense.llm.key_manager.time.sleep")
    def test_retries_with_backoff_when_all_rate_limited(self, mock_sleep):
        rotator = KeyRotator(api_keys=["a", "b"])
        rotator.mark_rate_limited("a")
        rotator.mark_rate_limited("b")

        # After 2 backoff sleeps, recover key "a"
        call_count = [0]
        def side_effect(seconds):
            call_count[0] += 1
            if call_count[0] == 2:
                rotator.mark_available("a")

        mock_sleep.side_effect = side_effect
        key = rotator.get_key_with_retry()
        assert key == "a"
        assert mock_sleep.call_count == 2

    @patch("codesense.llm.key_manager.time.sleep")
    def test_backoff_starts_at_1_second(self, mock_sleep):
        rotator = KeyRotator(api_keys=["a"])
        rotator.mark_rate_limited("a")

        # Recover after first sleep
        mock_sleep.side_effect = lambda _: rotator.mark_available("a")
        rotator.get_key_with_retry()
        mock_sleep.assert_called_with(1.0)

    @patch("codesense.llm.key_manager.time.sleep")
    def test_backoff_doubles_each_attempt(self, mock_sleep):
        rotator = KeyRotator(api_keys=["a"])
        rotator.mark_rate_limited("a")

        sleep_values = []
        call_count = [0]
        def side_effect(seconds):
            sleep_values.append(seconds)
            call_count[0] += 1
            if call_count[0] == 4:
                rotator.mark_available("a")

        mock_sleep.side_effect = side_effect
        rotator.get_key_with_retry()
        assert sleep_values == [1.0, 2.0, 4.0, 8.0]

    @patch("codesense.llm.key_manager.time.sleep")
    def test_backoff_capped_at_60_seconds(self, mock_sleep):
        rotator = KeyRotator(api_keys=["a"])
        rotator.mark_rate_limited("a")

        sleep_values = []
        def side_effect(seconds):
            sleep_values.append(seconds)

        mock_sleep.side_effect = side_effect
        with pytest.raises(RuntimeError, match="maximum retry attempts"):
            rotator.get_key_with_retry()

        # Backoff sequence: 1, 2, 4, 8, 16 (all under 60s cap for 5 retries)
        assert sleep_values == [1.0, 2.0, 4.0, 8.0, 16.0]

    @patch("codesense.llm.key_manager.time.sleep")
    def test_max_5_retries_then_raises_error(self, mock_sleep):
        rotator = KeyRotator(api_keys=["a", "b"])
        rotator.mark_rate_limited("a")
        rotator.mark_rate_limited("b")

        mock_sleep.side_effect = lambda _: None  # Don't recover any keys
        with pytest.raises(RuntimeError, match="maximum retry attempts.*5.*exhausted"):
            rotator.get_key_with_retry()
        assert mock_sleep.call_count == 5

    @patch("codesense.llm.key_manager.time.sleep")
    def test_recovery_during_backoff_returns_key(self, mock_sleep):
        """After backoff wait, if a key recovers past 60s cooldown, it should be returned."""
        rotator = KeyRotator(api_keys=["a", "b"])
        rotator.mark_rate_limited("a")
        rotator.mark_rate_limited("b")

        # Simulate key "a" recovering after first sleep (past 60s cooldown)
        def side_effect(seconds):
            rotator.pool.keys[0].rate_limited_at = datetime.now(timezone.utc) - timedelta(seconds=61)

        mock_sleep.side_effect = side_effect
        key = rotator.get_key_with_retry()
        assert key == "a"

    def test_get_next_key_still_works_independently(self):
        """Existing get_next_key() method should work unchanged."""
        rotator = KeyRotator(api_keys=["a", "b", "c"])
        keys = [rotator.get_next_key() for _ in range(6)]
        assert keys == ["a", "b", "c", "a", "b", "c"]


class TestAvailableKeys:
    """Test available_keys property."""

    def test_all_available_initially(self):
        rotator = KeyRotator(api_keys=["a", "b", "c"])
        assert rotator.available_keys == ["a", "b", "c"]

    def test_excludes_rate_limited(self):
        rotator = KeyRotator(api_keys=["a", "b", "c"])
        rotator.mark_rate_limited("b")
        assert rotator.available_keys == ["a", "c"]

    def test_includes_recovered_keys(self):
        rotator = KeyRotator(api_keys=["a", "b"])
        rotator.mark_rate_limited("a")
        rotator.pool.keys[0].rate_limited_at = datetime.now(timezone.utc) - timedelta(seconds=61)
        assert "a" in rotator.available_keys
