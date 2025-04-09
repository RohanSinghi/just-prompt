"""
Groq provider implementation for Just-Prompt
"""
import os
import time
from typing import List, Optional, Dict, Any

from groq import Groq, AsyncGroq
from groq.errors import RateLimitError, AuthenticationError, BadRequestError

from just_prompt.atoms.shared.data_types import PromptResponse


class GroqProvider:
    """Groq provider implementation"""
    
    def __init__(self):
        """Initialize the Groq provider with API key"""
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)
    
    async def list_models(self) -> List[str]:
        """List available models from Groq"""
        try:
            # Fetch models from Groq API
            models_response = await self.async_client.models.list()
            
            # Extract and return the model IDs
            model_ids = [model.id for model in models_response.data]
            return model_ids
        except Exception as e:
            return await self._handle_error(e)
        
    async def generate(self, prompt: str, model: str) -> PromptResponse:
        """Generate a response for the given prompt using the specified model"""
        try:
            # Prepare the chat completion request
            chat_completion = await self.async_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            
            # Extract the response text
            content = chat_completion.choices[0].message.content
            
            # Calculate token usage if available
            tokens = None
            if hasattr(chat_completion, 'usage'):
                tokens = chat_completion.usage.total_tokens
            
            return PromptResponse(
                model=model,
                content=content,
                tokens=tokens
            )
        except Exception as e:
            return await self._handle_error(e, retry_count=0, prompt=prompt, model=model)
    
    async def _handle_error(self, error: Exception, retry_count: int = 0, **kwargs) -> Any:
        """Handle errors with appropriate retry logic"""
        # Maximum number of retries
        max_retries = 3
        
        # Handle rate limiting errors
        if isinstance(error, RateLimitError) and retry_count < max_retries:
            # Exponential backoff: wait longer between each retry
            wait_time = 2 ** retry_count
            time.sleep(wait_time)
            
            # Extract prompt and model from kwargs if they exist
            prompt = kwargs.get("prompt")
            model = kwargs.get("model")
            
            if prompt and model:
                # Retry the generate method with incremented retry count
                retry_count += 1
                return await self.generate(prompt, model)
            else:
                # If we don't have enough information to retry, re-raise the error
                raise error
        
        # Handle authentication errors
        elif isinstance(error, AuthenticationError):
            raise ValueError(f"Groq API key is invalid: {str(error)}")
        
        # Handle bad request errors (invalid model, etc.)
        elif isinstance(error, BadRequestError):
            raise ValueError(f"Invalid request to Groq API: {str(error)}")
        
        # Handle API errors with retry for transient issues
        elif retry_count < max_retries and kwargs.get("prompt") and kwargs.get("model"):
            # Wait a bit and retry
            time.sleep(1)
            
            # Retry the generate method with incremented retry count
            retry_count += 1
            return await self.generate(kwargs["prompt"], kwargs["model"])
        
        # Handle other errors
        else:
            raise ValueError(f"Error occurred when calling Groq API: {str(error)}") 