"""Unit tests for GeminiService."""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from codesense.llm.key_manager import KeyRotator
from codesense.llm.gemini_service import GeminiService


class TestGeminiServiceInit:
    """Test GeminiService initialization."""

    def test_init_with_defaults(self):
        rotator = KeyRotator(api_keys=["key-1"])
        service = GeminiService(key_rotator=rotator)
        assert service._model == "gemini-1.5-flash"
        assert service._timeout == 30
        assert service._max_retries == 5

    def test_init_with_custom_model(self):
        rotator = KeyRotator(api_keys=["key-1"])
        service = GeminiService(key_rotator=rotator, model="gemini-2.0-flash")
        assert service._model == "gemini-2.0-flash"

    def test_init_with_custom_timeout(self):
        rotator = KeyRotator(api_keys=["key-1"])
        service = GeminiService(key_rotator=rotator, timeout=60)
        assert service._timeout == 60


class TestGenerate:
    """Test the generate method."""

    @patch("codesense.llm.gemini_service._get_chat_google_generative_ai")
    def test_successful_generation(self, mock_get_llm_class):
        """Test that generate returns LLM response on success."""
        rotator = KeyRotator(api_keys=["key-1", "key-2"])
        service = GeminiService(key_rotator=rotator)

        # Mock the LLM to return a response via LCEL chain
        mock_llm_class = MagicMock()
        mock_get_llm_class.return_value = mock_llm_class
        mock_llm = MagicMock()
        mock_llm.__or__ = MagicMock(return_value=MagicMock())
        mock_llm_class.return_value = mock_llm

        # Mock the full chain invocation
        with patch.object(service, '_create_llm') as mock_create:
            mock_chain_result = "Generated response"
            mock_model = MagicMock()
            mock_create.return_value = mock_model
            # The chain is: prompt_template | llm | output_parser
            # We need to mock the pipe operator chain result
            mock_model.__or__ = MagicMock()
            chain_mock = MagicMock()
            chain_mock.invoke.return_value = mock_chain_result

            with patch("langchain_core.prompts.ChatPromptTemplate.from_messages") as mock_from_msgs:
                mock_prompt_instance = MagicMock()
                mock_from_msgs.return_value = mock_prompt_instance
                mock_prompt_instance.__or__ = MagicMock(return_value=MagicMock())
                mock_prompt_instance.__or__.return_value.__or__ = MagicMock(return_value=chain_mock)

                result = service.generate("test prompt")
                assert result == mock_chain_result

    @patch("codesense.llm.gemini_service._get_chat_google_generative_ai")
    def test_rate_limit_triggers_key_rotation(self, mock_get_llm_class):
        """Test that rate-limit error marks key and retries with next key."""
        rotator = KeyRotator(api_keys=["key-1", "key-2", "key-3"])
        service = GeminiService(key_rotator=rotator)

        mock_llm_class = MagicMock()
        mock_get_llm_class.return_value = mock_llm_class

        call_count = [0]

        def invoke_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("429 Resource Exhausted: rate limit exceeded")
            return "success on second key"

        chain_mock = MagicMock()
        chain_mock.invoke.side_effect = invoke_side_effect

        with patch("langchain_core.prompts.ChatPromptTemplate.from_messages") as mock_from_msgs:
            mock_prompt_instance = MagicMock()
            mock_from_msgs.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = MagicMock(return_value=MagicMock())
            mock_prompt_instance.__or__.return_value.__or__ = MagicMock(return_value=chain_mock)

            result = service.generate("test prompt")
            assert result == "success on second key"
            # First key should be marked rate-limited
            assert rotator.pool.keys[0].is_available is False

    @patch("codesense.llm.gemini_service._get_chat_google_generative_ai")
    def test_non_rate_limit_error_raised_immediately(self, mock_get_llm_class):
        """Test that non-rate-limit errors are re-raised immediately."""
        rotator = KeyRotator(api_keys=["key-1", "key-2"])
        service = GeminiService(key_rotator=rotator)

        mock_llm_class = MagicMock()
        mock_get_llm_class.return_value = mock_llm_class

        chain_mock = MagicMock()
        chain_mock.invoke.side_effect = ValueError("Invalid input format")

        with patch("langchain_core.prompts.ChatPromptTemplate.from_messages") as mock_from_msgs:
            mock_prompt_instance = MagicMock()
            mock_from_msgs.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = MagicMock(return_value=MagicMock())
            mock_prompt_instance.__or__.return_value.__or__ = MagicMock(return_value=chain_mock)

            with pytest.raises(ValueError, match="Invalid input format"):
                service.generate("test prompt")

    @patch("codesense.llm.gemini_service._get_chat_google_generative_ai")
    def test_timeout_error_raised_immediately(self, mock_get_llm_class):
        """Test that timeout errors are raised immediately without retrying."""
        rotator = KeyRotator(api_keys=["key-1", "key-2"])
        service = GeminiService(key_rotator=rotator, timeout=30)

        mock_llm_class = MagicMock()
        mock_get_llm_class.return_value = mock_llm_class

        chain_mock = MagicMock()
        chain_mock.invoke.side_effect = TimeoutError("Request timed out after 30 seconds")

        with patch("langchain_core.prompts.ChatPromptTemplate.from_messages") as mock_from_msgs:
            mock_prompt_instance = MagicMock()
            mock_from_msgs.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = MagicMock(return_value=MagicMock())
            mock_prompt_instance.__or__.return_value.__or__ = MagicMock(return_value=chain_mock)

            with pytest.raises(TimeoutError, match="Request timed out"):
                service.generate("test prompt")
            # Verify it didn't try to rotate keys - only 1 invocation
            assert chain_mock.invoke.call_count == 1

    @patch("codesense.llm.gemini_service.time.sleep")
    @patch("codesense.llm.gemini_service._get_chat_google_generative_ai")
    def test_all_keys_exhausted_raises_runtime_error(self, mock_get_llm_class, mock_sleep):
        """Test that RuntimeError is raised when all retries are exhausted."""
        rotator = KeyRotator(api_keys=["key-1"])
        service = GeminiService(key_rotator=rotator, max_retries=3)

        mock_llm_class = MagicMock()
        mock_get_llm_class.return_value = mock_llm_class

        chain_mock = MagicMock()
        chain_mock.invoke.side_effect = Exception("429 rate limit exceeded")

        with patch("langchain_core.prompts.ChatPromptTemplate.from_messages") as mock_from_msgs:
            mock_prompt_instance = MagicMock()
            mock_from_msgs.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = MagicMock(return_value=MagicMock())
            mock_prompt_instance.__or__.return_value.__or__ = MagicMock(return_value=chain_mock)

            with pytest.raises(RuntimeError, match="rate-limited"):
                service.generate("test prompt")


class TestGenerateStructured:
    """Test the generate_structured method."""

    @patch("codesense.llm.gemini_service._get_chat_google_generative_ai")
    def test_structured_generation_with_parser(self, mock_get_llm_class):
        """Test that generate_structured uses the output parser correctly."""
        rotator = KeyRotator(api_keys=["key-1"])
        service = GeminiService(key_rotator=rotator)

        mock_llm_class = MagicMock()
        mock_get_llm_class.return_value = mock_llm_class

        mock_parser = MagicMock()
        mock_parser.get_format_instructions.return_value = "Return JSON with field 'name'"

        expected_result = {"name": "test"}
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = expected_result

        with patch("langchain_core.prompts.ChatPromptTemplate.from_messages") as mock_from_msgs:
            mock_prompt_instance = MagicMock()
            mock_from_msgs.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = MagicMock(return_value=MagicMock())
            mock_prompt_instance.__or__.return_value.__or__ = MagicMock(return_value=chain_mock)

            result = service.generate_structured("Extract the name", mock_parser)
            assert result == expected_result

    @patch("codesense.llm.gemini_service._get_chat_google_generative_ai")
    def test_structured_generation_rate_limit_retry(self, mock_get_llm_class):
        """Test that generate_structured retries on rate-limit errors."""
        rotator = KeyRotator(api_keys=["key-1", "key-2"])
        service = GeminiService(key_rotator=rotator)

        mock_llm_class = MagicMock()
        mock_get_llm_class.return_value = mock_llm_class

        mock_parser = MagicMock()
        mock_parser.get_format_instructions.return_value = "Return JSON"

        call_count = [0]

        def invoke_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("ResourceExhausted: quota exceeded")
            return {"result": "success"}

        chain_mock = MagicMock()
        chain_mock.invoke.side_effect = invoke_side_effect

        with patch("langchain_core.prompts.ChatPromptTemplate.from_messages") as mock_from_msgs:
            mock_prompt_instance = MagicMock()
            mock_from_msgs.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = MagicMock(return_value=MagicMock())
            mock_prompt_instance.__or__.return_value.__or__ = MagicMock(return_value=chain_mock)

            result = service.generate_structured("test", mock_parser)
            assert result == {"result": "success"}


class TestRateLimitDetection:
    """Test _is_rate_limit_error static method."""

    def test_detects_429_error(self):
        error = Exception("HTTP 429: Too Many Requests")
        assert GeminiService._is_rate_limit_error(error) is True

    def test_detects_resource_exhausted(self):
        error = Exception("google.api_core.exceptions.ResourceExhausted: resource exhausted")
        assert GeminiService._is_rate_limit_error(error) is True

    def test_detects_rate_limit_text(self):
        error = Exception("Rate limit exceeded for this API key")
        assert GeminiService._is_rate_limit_error(error) is True

    def test_detects_quota_error(self):
        error = Exception("Quota exceeded for this project")
        assert GeminiService._is_rate_limit_error(error) is True

    def test_non_rate_limit_error_returns_false(self):
        error = ValueError("Invalid argument")
        assert GeminiService._is_rate_limit_error(error) is False

    def test_connection_error_returns_false(self):
        error = ConnectionError("Network unreachable")
        assert GeminiService._is_rate_limit_error(error) is False


class TestBackoffCalculation:
    """Test _get_backoff_seconds method."""

    def test_backoff_attempt_0(self):
        rotator = KeyRotator(api_keys=["key-1"])
        service = GeminiService(key_rotator=rotator)
        assert service._get_backoff_seconds(0) == 1.0

    def test_backoff_attempt_1(self):
        rotator = KeyRotator(api_keys=["key-1"])
        service = GeminiService(key_rotator=rotator)
        assert service._get_backoff_seconds(1) == 2.0

    def test_backoff_attempt_2(self):
        rotator = KeyRotator(api_keys=["key-1"])
        service = GeminiService(key_rotator=rotator)
        assert service._get_backoff_seconds(2) == 4.0

    def test_backoff_attempt_3(self):
        rotator = KeyRotator(api_keys=["key-1"])
        service = GeminiService(key_rotator=rotator)
        assert service._get_backoff_seconds(3) == 8.0

    def test_backoff_attempt_4(self):
        rotator = KeyRotator(api_keys=["key-1"])
        service = GeminiService(key_rotator=rotator)
        assert service._get_backoff_seconds(4) == 16.0

    def test_backoff_capped_at_60(self):
        rotator = KeyRotator(api_keys=["key-1"])
        service = GeminiService(key_rotator=rotator)
        assert service._get_backoff_seconds(6) == 60.0
        assert service._get_backoff_seconds(10) == 60.0


class TestCreateLLM:
    """Test _create_llm method."""

    @patch("codesense.llm.gemini_service._get_chat_google_generative_ai")
    def test_creates_llm_with_correct_params(self, mock_get_llm_class):
        """Test that _create_llm passes the right configuration."""
        mock_llm_class = MagicMock()
        mock_get_llm_class.return_value = mock_llm_class

        rotator = KeyRotator(api_keys=["key-1"])
        service = GeminiService(key_rotator=rotator, model="gemini-2.0-flash", timeout=45)

        service._create_llm("test-api-key")

        mock_llm_class.assert_called_once_with(
            model="gemini-2.0-flash",
            google_api_key="test-api-key",
            timeout=45,
            max_retries=0,
        )
