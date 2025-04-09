"""
Tests for the Groq provider
"""
import os
import pytest
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch

from groq.errors import RateLimitError, AuthenticationError, BadRequestError

from just_prompt.atoms.llm_providers.groq import GroqProvider
from just_prompt.atoms.shared.data_types import PromptResponse


class TestGroqProvider:
    """Tests for the Groq provider"""

    @mock.patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
    @mock.patch("groq.Groq")
    @mock.patch("groq.AsyncGroq")
    def test_init(self, mock_async_groq, mock_groq):
        """Test initialization"""
        provider = GroqProvider()
        assert provider.api_key == "test_key"
        mock_groq.assert_called_once_with(api_key="test_key")
        mock_async_groq.assert_called_once_with(api_key="test_key")

    @mock.patch.dict(os.environ, {})
    def test_init_missing_key(self):
        """Test initialization with missing API key"""
        with pytest.raises(ValueError, match="GROQ_API_KEY environment variable not set"):
            GroqProvider()

    @mock.patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
    @mock.patch("groq.Groq")
    @mock.patch("groq.AsyncGroq")
    async def test_list_models(self, mock_async_groq, mock_groq):
        """Test listing models"""
        # Setup mock models
        mock_model1 = MagicMock()
        mock_model1.id = "llama2-70b-4096"
        
        mock_model2 = MagicMock()
        mock_model2.id = "mixtral-8x7b-32768"
        
        # Setup mock response
        mock_models_response = MagicMock()
        mock_models_response.data = [mock_model1, mock_model2]
        
        # Setup mock client
        mock_client_instance = MagicMock()
        mock_client_instance.models.list = AsyncMock(return_value=mock_models_response)
        mock_async_groq.return_value = mock_client_instance
        
        # Initialize provider and call list_models
        provider = GroqProvider()
        models = await provider.list_models()
        
        # Check that we got the expected response
        assert "llama2-70b-4096" in models
        assert "mixtral-8x7b-32768" in models
        assert len(models) == 2

    @mock.patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
    @mock.patch("groq.Groq")
    @mock.patch("groq.AsyncGroq")
    async def test_generate(self, mock_async_groq, mock_groq):
        """Test generating a response"""
        # Setup chat completion mock
        mock_message = MagicMock()
        mock_message.content = "Test response"
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_usage = MagicMock()
        mock_usage.total_tokens = 42
        
        mock_chat_completion = MagicMock()
        mock_chat_completion.choices = [mock_choice]
        mock_chat_completion.usage = mock_usage
        
        # Setup mock client
        mock_client_instance = MagicMock()
        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(return_value=mock_chat_completion)
        mock_client_instance.chat.completions = mock_completions
        mock_async_groq.return_value = mock_client_instance
        
        # Initialize provider and call generate
        provider = GroqProvider()
        response = await provider.generate("Test prompt", "llama2-70b-4096")
        
        # Check that we called create with the right parameters
        mock_completions.create.assert_called_once_with(
            model="llama2-70b-4096",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.7,
            max_tokens=1024,
        )
        
        # Check that we got the expected response
        assert isinstance(response, PromptResponse)
        assert response.model == "llama2-70b-4096"
        assert response.content == "Test response"
        assert response.tokens == 42

    @mock.patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
    @mock.patch("groq.Groq")
    @mock.patch("groq.AsyncGroq")
    @mock.patch("time.sleep")
    async def test_handle_rate_limit_error(self, mock_sleep, mock_async_groq, mock_groq):
        """Test handling rate limit errors"""
        # Setup mock client
        mock_client_instance = MagicMock()
        mock_completions = MagicMock()
        
        # First call raises a rate limit error, second call succeeds
        mock_message = MagicMock()
        mock_message.content = "Retry response"
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_chat_completion = MagicMock()
        mock_chat_completion.choices = [mock_choice]
        
        mock_completions.create = AsyncMock(
            side_effect=[
                RateLimitError("Rate limit exceeded"),
                mock_chat_completion
            ]
        )
        
        mock_client_instance.chat.completions = mock_completions
        mock_async_groq.return_value = mock_client_instance
        
        # Initialize provider and call generate
        provider = GroqProvider()
        response = await provider.generate("Test prompt", "llama2-70b-4096")
        
        # Check that sleep was called for exponential backoff
        mock_sleep.assert_called_once_with(1)  # First retry = 2^0 = 1 second
        
        # Check that we called create twice (one error, one success)
        assert mock_completions.create.call_count == 2
        
        # Check that we got the expected response after retry
        assert response.model == "llama2-70b-4096"
        assert response.content == "Retry response"

    @mock.patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
    @mock.patch("groq.Groq")
    @mock.patch("groq.AsyncGroq")
    async def test_handle_authentication_error(self, mock_async_groq, mock_groq):
        """Test handling authentication errors"""
        # Setup mock client
        mock_client_instance = MagicMock()
        mock_completions = MagicMock()
        
        # Raise an authentication error
        mock_completions.create = AsyncMock(side_effect=AuthenticationError("Invalid API key"))
        
        mock_client_instance.chat.completions = mock_completions
        mock_async_groq.return_value = mock_client_instance
        
        # Initialize provider and call generate
        provider = GroqProvider()
        
        # Check that we raise the expected error
        with pytest.raises(ValueError, match="Groq API key is invalid"):
            await provider.generate("Test prompt", "llama2-70b-4096")

    @mock.patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
    @mock.patch("groq.Groq")
    @mock.patch("groq.AsyncGroq")
    async def test_handle_bad_request_error(self, mock_async_groq, mock_groq):
        """Test handling bad request errors"""
        # Setup mock client
        mock_client_instance = MagicMock()
        mock_completions = MagicMock()
        
        # Raise a bad request error
        mock_completions.create = AsyncMock(side_effect=BadRequestError("Invalid model name"))
        
        mock_client_instance.chat.completions = mock_completions
        mock_async_groq.return_value = mock_client_instance
        
        # Initialize provider and call generate
        provider = GroqProvider()
        
        # Check that we raise the expected error
        with pytest.raises(ValueError, match="Invalid request to Groq API"):
            await provider.generate("Test prompt", "invalid-model") 