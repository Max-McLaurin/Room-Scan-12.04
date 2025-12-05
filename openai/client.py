"""
OpenAI API client wrapper for consistent usage across the application.

This module provides a centralized client for OpenAI API interactions with:
- Automatic retry logic for transient failures (up to 3 attempts with exponential backoff)
- Rate limit handling
- Comprehensive logging
- Consistent error handling
- Support for JSON responses and vision/image-based prompts
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from django.conf import settings
from openai import OpenAI
from openai.types.chat import ChatCompletion

from .exceptions import (
    OpenAIError,
    OpenAIRateLimitError,
    OpenAIAPIConnectionError,
    OpenAIAPIResponseError
)

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Centralized OpenAI API client with retry logic and error handling.
    
    This client wraps the OpenAI Python SDK and provides:
    - Consistent error handling
    - Automatic retries for transient failures
    - Rate limit handling
    - Logging of all API calls
    - Support for JSON response format
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key (defaults to settings.OPENAI_API_KEY)
            model: Default model to use (defaults to settings.OPENAI_MODEL or 'gpt-4o-mini')
        """
        self.api_key = api_key or getattr(settings, 'OPENAI_API_KEY', None)
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either as parameter or in settings.OPENAI_API_KEY")
        
        self.model = model or getattr(settings, 'OPENAI_MODEL', 'gpt-4o-mini')
        self.client = OpenAI(api_key=self.api_key)
        
        logger.info(f"Initialized OpenAI client with model: {self.model}")
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if the error is a rate limit error."""
        error_str = str(error).lower()
        return (
            'rate limit' in error_str or
            'too many requests' in error_str or
            '429' in error_str
        )
    
    def _log_api_call(self, method: str, **kwargs):
        """Log API call details."""
        logger.info(f"OpenAI API call: {method} with model={self.model}")
        if 'messages' in kwargs:
            logger.debug(f"Number of messages: {len(kwargs['messages'])}")
        if 'max_tokens' in kwargs:
            logger.debug(f"Max tokens: {kwargs['max_tokens']}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        **kwargs
    ) -> ChatCompletion:
        """
        Create a chat completion with automatic retry logic.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to instance default)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            response_format: Response format specification (e.g., {"type": "json_object"})
            max_retries: Maximum number of retry attempts (default: 3)
            **kwargs: Additional arguments to pass to OpenAI API
        
        Returns:
            ChatCompletion object
            
        Raises:
            OpenAIRateLimitError: If rate limit is exceeded
            OpenAIAPIConnectionError: If there's a connection error
            OpenAIAPIResponseError: If the API returns an unexpected response
        """
        model = model or self.model
        
        self._log_api_call(
            'chat_completion',
            model=model,
            max_tokens=max_tokens,
            messages=messages
        )
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                    **kwargs
                )
                
                logger.info(f"OpenAI API call successful")
                return response
                
            except Exception as e:
                last_exception = e
                logger.error(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {e}")
                
                # Don't retry on rate limit errors
                if self._is_rate_limit_error(e):
                    raise OpenAIRateLimitError(f"OpenAI rate limit exceeded: {e}")
                
                # Don't retry on last attempt
                if attempt == max_retries - 1:
                    break
                
                # Wait before retrying (exponential backoff)
                wait_time = 2 ** attempt
                logger.warning(f"Retrying OpenAI API call in {wait_time}s...")
                time.sleep(wait_time)
        
        # If we get here, all retries failed
        if 'connection' in str(last_exception).lower() or 'timeout' in str(last_exception).lower():
            raise OpenAIAPIConnectionError(f"OpenAI connection error: {last_exception}")
        
        raise OpenAIError(f"OpenAI API error after {max_retries} attempts: {last_exception}")
    
    def chat_completion_json(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion and return parsed JSON response.
        
        Args:
            messages: List of message dictionaries
            model: Model to use
            max_tokens: Maximum tokens in response
            **kwargs: Additional arguments
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            OpenAIAPIResponseError: If JSON parsing fails
        """
        response = self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            **kwargs
        )
        
        try:
            content = response.choices[0].message.content
            if not content:
                raise OpenAIAPIResponseError("Empty response from OpenAI")
            
            return json.loads(content)
            
        except (IndexError, AttributeError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse OpenAI JSON response: {e}")
            raise OpenAIAPIResponseError(f"Failed to parse OpenAI response: {e}")
    
    def extract_json_from_response(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion and extract JSON from the response content.
        
        This method handles cases where JSON might be embedded in text.
        
        Args:
            messages: List of message dictionaries
            model: Model to use
            max_tokens: Maximum tokens in response
            **kwargs: Additional arguments
            
        Returns:
            Parsed JSON dictionary
        """
        import re
        
        response = self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            **kwargs
        )
        
        try:
            content = response.choices[0].message.content
            if not content:
                raise OpenAIAPIResponseError("Empty response from OpenAI")
            
            # Try direct JSON parsing first
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Extract JSON from text using regex
                json_match = re.search(r"(\{.*\}|\[.*\])", content, re.DOTALL)
                if not json_match:
                    raise OpenAIAPIResponseError("No JSON data found in response")
                
                return json.loads(json_match.group(0))
                
        except (IndexError, AttributeError, json.JSONDecodeError) as e:
            logger.error(f"Failed to extract JSON from OpenAI response: {e}")
            raise OpenAIAPIResponseError(f"Failed to extract JSON from response: {e}")
    
    def create_with_vision(
        self,
        text_prompt: str,
        images: List[str],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatCompletion:
        """
        Create a chat completion with vision (image inputs).
        
        Args:
            text_prompt: Text prompt for the model
            images: List of base64-encoded images
            model: Model to use (defaults to instance default)
            max_tokens: Maximum tokens in response
            **kwargs: Additional arguments
            
        Returns:
            ChatCompletion object
        """
        content_list = [{"type": "text", "text": text_prompt}]
        
        for image in images:
            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            })
        
        return self.chat_completion(
            messages=[{"role": "user", "content": content_list}],
            model=model,
            max_tokens=max_tokens,
            **kwargs
        )


# Convenience function to get a singleton instance
_client_instance: Optional[OpenAIClient] = None


def get_openai_client() -> OpenAIClient:
    """
    Get or create a singleton OpenAI client instance.
    
    Returns:
        OpenAIClient instance
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = OpenAIClient()
    return _client_instance

