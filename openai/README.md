OpenAI Client Module
This module provides a centralized wrapper for OpenAI API interactions across the application.

Features
Automatic Retry Logic: Handles transient failures with exponential backoff (up to 3 retries)
Rate Limit Handling: Detects and handles rate limit errors gracefully
Comprehensive Logging: Logs all API calls for debugging and monitoring
Consistent Error Handling: Custom exceptions for different error types
JSON Response Support: Built-in methods for JSON responses
Vision Support: Helper methods for image-based prompts
Usage
Basic Usage
from property_hero.openai import get_openai_client

# Get singleton client instance
client = get_openai_client()

# Create a chat completion
response = client.chat_completion(
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=100
)

# Extract response content
answer = response.choices[0].message.content
JSON Responses
from property_hero.openai import get_openai_client

client = get_openai_client()

# Get structured JSON response
data = client.chat_completion_json(
    messages=[
        {"role": "user", "content": "List 3 cities in JSON format"}
    ],
    max_tokens=200
)
With Vision (Images)
from property_hero.openai import get_openai_client
from property_hero.core.utils import encode_image_to_base64

client = get_openai_client()

# Encode image to base64
base64_image = encode_image_to_base64(image_file)

# Create completion with vision
response = client.create_with_vision(
    text_prompt="Describe what you see in this image",
    images=[base64_image],
    max_tokens=300
)
Error Handling
from property_hero.openai import (
    get_openai_client,
    OpenAIRateLimitError,
    OpenAIAPIConnectionError,
    OpenAIError
)

client = get_openai_client()

try:
    response = client.chat_completion(messages=[...])
except OpenAIRateLimitError:
    # Handle rate limit
    print("Rate limit exceeded, please try again later")
except OpenAIAPIConnectionError:
    # Handle connection errors
    print("Connection failed, check your network")
except OpenAIError as e:
    # Handle other OpenAI errors
    print(f"OpenAI error: {e}")
Configuration
The client reads configuration from Django settings:

OPENAI_API_KEY: Your OpenAI API key (required)
OPENAI_MODEL: Default model to use (defaults to 'gpt-4o-mini')
You can also instantiate the client with custom settings:

from property_hero.openai import OpenAIClient

# Custom instance
client = OpenAIClient(
    api_key="your-api-key",
    model="gpt-4o"
)
Migration Guide
Before (using direct OpenAI client)
from openai import OpenAI
from django.conf import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100
)
After (using centralized client)
from property_hero.openai import get_openai_client

client = get_openai_client()

response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100
)
Benefits
Consistency: All OpenAI calls use the same client and settings
Reliability: Automatic retries handle transient failures
Monitoring: Centralized logging makes it easy to track API usage
Error Handling: Consistent error handling across the application
Testing: Easy to mock the client for unit tests
Rate Limiting: Better handling of rate limit errors