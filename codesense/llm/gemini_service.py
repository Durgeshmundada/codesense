"""Gemini LLM service with key rotation integration.

Provides a sync interface to Google's Gemini model via LangChain,
with automatic key rotation and retry on rate-limit errors.
"""

from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

from codesense.llm.key_manager import KeyRotator

if TYPE_CHECKING:
    from langchain_google_genai import ChatGoogleGenerativeAI as _ChatGoogleGenerativeAI


def _get_chat_prompt_template():
    """Lazily import ChatPromptTemplate to avoid heavy module loading at import time."""
    from langchain_core.prompts import ChatPromptTemplate
    return ChatPromptTemplate


def _get_str_output_parser():
    """Lazily import StrOutputParser to avoid heavy module loading at import time."""
    from langchain_core.output_parsers import StrOutputParser
    return StrOutputParser


def _get_chat_google_generative_ai():
    """Lazily import ChatGoogleGenerativeAI to avoid heavy module loading at import time."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI


class GeminiService:
    """LLM service wrapping Gemini with automatic key rotation on rate limits.

    Uses KeyRotator to select API keys in round-robin order. When a rate-limit
    error occurs (ResourceExhausted / HTTP 429), the current key is marked as
    rate-limited and the next available key is tried.

    Args:
        key_rotator: KeyRotator instance managing the API key pool.
        model: Gemini model name to use. Defaults to "gemini-2.5-flash".
        timeout: Timeout in seconds per LLM call. Defaults to 30.
        max_retries: Maximum number of retry attempts on rate-limit errors.
            Defaults to 5 (matching the RotationPool max_retries).
    """

    def __init__(
        self,
        key_rotator: KeyRotator,
        model: str = "gemini-2.5-flash",
        timeout: int = 30,
        max_retries: int = 5,
    ) -> None:
        self._key_rotator = key_rotator
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

    def _create_llm(self, api_key: str) -> Any:
        """Create a ChatGoogleGenerativeAI instance with the given API key.

        Args:
            api_key: The Gemini API key to authenticate with.

        Returns:
            A configured ChatGoogleGenerativeAI instance.
        """
        ChatGoogleGenerativeAI = _get_chat_google_generative_ai()

        return ChatGoogleGenerativeAI(
            model=self._model,
            google_api_key=api_key,
            timeout=self._timeout,
            max_retries=0,  # We handle retries ourselves via key rotation
        )

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        """Check if an exception is a rate-limit error (ResourceExhausted / 429).

        Args:
            error: The exception to inspect.

        Returns:
            True if this is a rate-limit error that should trigger key rotation.
        """
        error_str = str(error).lower()
        # Check for Google API rate-limit indicators
        if "429" in str(error) or "resource exhausted" in error_str:
            return True
        if "resourceexhausted" in error_str:
            return True
        if "rate limit" in error_str or "quota" in error_str:
            return True
        # Check exception type name for google-specific errors
        error_type = type(error).__name__.lower()
        if "resourceexhausted" in error_type:
            return True
        return False

    @staticmethod
    def _is_auth_error(error: Exception) -> bool:
        """Check if an exception is an auth/permission error for the API key.

        These indicate a permanently unusable key (revoked, invalid, or a
        project denied access) rather than a transient rate limit. Such keys
        should be disabled and skipped, not retried.

        Args:
            error: The exception to inspect.

        Returns:
            True if this is an authentication/permission error.
        """
        error_str = str(error).lower()
        if "403" in str(error) or "401" in str(error):
            return True
        if "permission_denied" in error_str or "permission denied" in error_str:
            return True
        if "denied access" in error_str or "unauthenticated" in error_str:
            return True
        if "api key not valid" in error_str or "api_key_invalid" in error_str:
            return True
        error_type = type(error).__name__.lower()
        if "permissiondenied" in error_type or "unauthenticated" in error_type:
            return True
        return False

    def _get_backoff_seconds(self, attempt: int) -> float:
        """Calculate exponential backoff duration for a retry attempt.

        Args:
            attempt: Zero-based retry attempt number.

        Returns:
            Backoff duration in seconds: min(2^attempt, 60).
        """
        return min(2**attempt, 60.0)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a text response from the Gemini model.

        Uses LCEL pattern: prompt_template | llm | StrOutputParser.
        On rate-limit errors, marks the current key as rate-limited and retries
        with the next available key from the rotation pool.

        Args:
            prompt: The prompt text to send to the model.
            **kwargs: Additional keyword arguments passed to the LLM invocation.

        Returns:
            The generated response text.

        Raises:
            RuntimeError: If all retry attempts are exhausted with rate-limit errors.
            Exception: Any non-rate-limit error from the LLM is re-raised.
        """
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        # Sanitize prompt: replace non-ASCII chars that cause encoding errors
        prompt = prompt.encode("ascii", errors="replace").decode("ascii")

        prompt_template = ChatPromptTemplate.from_messages(
            [("human", "{input}")]
        )
        output_parser = StrOutputParser()

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                api_key = self._key_rotator.get_next_key()
            except RuntimeError:
                # All keys are rate-limited; apply backoff and retry
                if attempt < self._max_retries:
                    backoff = self._get_backoff_seconds(attempt)
                    time.sleep(backoff)
                    continue
                else:
                    raise RuntimeError(
                        "All API keys are rate-limited and maximum retry attempts exhausted. "
                        "Please wait for cooldown period to expire or add more keys."
                    )

            try:
                llm = self._create_llm(api_key)
                chain = prompt_template | llm | output_parser
                result = chain.invoke({"input": prompt, **kwargs})
                return result
            except Exception as e:
                if self._is_rate_limit_error(e):
                    self._key_rotator.mark_rate_limited(api_key)
                    last_error = e
                    # If we still have retries left, try next key
                    if attempt < self._max_retries:
                        # If all keys might be exhausted, backoff before retry
                        if self._key_rotator.is_all_rate_limited():
                            backoff = self._get_backoff_seconds(attempt)
                            time.sleep(backoff)
                        continue
                    else:
                        raise RuntimeError(
                            "All API keys are rate-limited and maximum retry attempts exhausted. "
                            f"Last error: {e}"
                        )
                else:
                    # Non-rate-limit errors are raised immediately
                    if self._is_auth_error(e):
                        # Permanently disable the denied/invalid key and rotate.
                        self._key_rotator.mark_invalid(api_key)
                        last_error = e
                        if (
                            attempt < self._max_retries
                            and self._key_rotator.has_usable_keys()
                        ):
                            continue
                        raise RuntimeError(
                            "All configured Gemini API keys are invalid or denied "
                            f"access. Last error: {e}"
                        )
                    raise

        # Should not reach here, but guard against it
        raise RuntimeError(
            f"Failed to generate response after {self._max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    def generate_structured(self, prompt: str, output_parser: Any) -> dict:
        """Generate a structured response using a LangChain output parser.

        Uses LCEL pattern: prompt_template | llm | output_parser.
        The output parser determines the structure of the returned data
        (e.g., PydanticOutputParser, JsonOutputParser).

        On rate-limit errors, marks the current key as rate-limited and retries
        with the next available key from the rotation pool.

        Args:
            prompt: The prompt text to send to the model.
            output_parser: A LangChain output parser instance that defines
                the expected output structure (e.g., PydanticOutputParser).

        Returns:
            The parsed structured response as a dictionary (or Pydantic model,
            depending on the parser).

        Raises:
            RuntimeError: If all retry attempts are exhausted with rate-limit errors.
            Exception: Any non-rate-limit error from the LLM is re-raised.
        """
        from langchain_core.prompts import ChatPromptTemplate

        # Sanitize prompt: replace non-ASCII chars that cause encoding errors
        prompt = prompt.encode("ascii", errors="replace").decode("ascii")

        # Build format instructions into the prompt if the parser supports it
        format_instructions = ""
        if hasattr(output_parser, "get_format_instructions"):
            format_instructions = output_parser.get_format_instructions()

        prompt_template = ChatPromptTemplate.from_messages(
            [("human", "{input}\n\n{format_instructions}")]
        )

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                api_key = self._key_rotator.get_next_key()
            except RuntimeError:
                # All keys are rate-limited; apply backoff and retry
                if attempt < self._max_retries:
                    backoff = self._get_backoff_seconds(attempt)
                    time.sleep(backoff)
                    continue
                else:
                    raise RuntimeError(
                        "All API keys are rate-limited and maximum retry attempts exhausted. "
                        "Please wait for cooldown period to expire or add more keys."
                    )

            try:
                llm = self._create_llm(api_key)
                chain = prompt_template | llm | output_parser
                result = chain.invoke({
                    "input": prompt,
                    "format_instructions": format_instructions,
                })
                return result
            except Exception as e:
                if self._is_rate_limit_error(e):
                    self._key_rotator.mark_rate_limited(api_key)
                    last_error = e
                    if attempt < self._max_retries:
                        if self._key_rotator.is_all_rate_limited():
                            backoff = self._get_backoff_seconds(attempt)
                            time.sleep(backoff)
                        continue
                    else:
                        raise RuntimeError(
                            "All API keys are rate-limited and maximum retry attempts exhausted. "
                            f"Last error: {e}"
                        )
                else:
                    if self._is_auth_error(e):
                        # Permanently disable the denied/invalid key and rotate.
                        self._key_rotator.mark_invalid(api_key)
                        last_error = e
                        if (
                            attempt < self._max_retries
                            and self._key_rotator.has_usable_keys()
                        ):
                            continue
                        raise RuntimeError(
                            "All configured Gemini API keys are invalid or denied "
                            f"access. Last error: {e}"
                        )
                    raise

        raise RuntimeError(
            f"Failed to generate structured response after {self._max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )
