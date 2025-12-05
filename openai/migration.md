noventum/property-hero-django

Skip to:

Main
Noventum
Property Hero
Property Hero Django
MIGRATION.md

Pull requests

Check out


Source

source:staging


ff7902a

Full commit
property-hero-django/
property_hero/
openai/
MIGRATION.md

Edit

Migration Guide: Centralized OpenAI Client
This guide explains how to migrate from direct OpenAI API calls to the centralized client.

Why Migrate?
Problems with Current Approach
Inconsistent Patterns: Different files use different approaches (OpenAI SDK vs direct HTTP)
No Retry Logic: Transient failures cause permanent errors
No Rate Limit Handling: Rate limit errors aren't handled gracefully
Difficult to Monitor: No centralized logging of API calls
Hard to Test: Module-level client instances are difficult to mock
Configuration Issues: Each file manages its own configuration
Benefits of Centralized Client
✅ Consistent API Usage: Same pattern everywhere
✅ Automatic Retries: Exponential backoff for transient failures
✅ Rate Limit Handling: Graceful degradation on rate limits
✅ Centralized Logging: Easy to monitor and debug
✅ Easy Testing: Simple to mock for unit tests
✅ Single Configuration: Configure once, use everywhere
Migration Steps
1. Install Dependencies
pip install -r requirements/base.txt
All required dependencies are already included (no new packages needed).

2. Update Django Settings
Already done: OPENAI_MODEL has been added to config/settings/base.py

3. Replace Direct Client Usage
Before
# property_hero/assets/utils.py (OLD)
from openai import OpenAI
from django.conf import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100
)
After
# property_hero/assets/utils.py (NEW)
from property_hero.openai import get_openai_client

client = get_openai_client()

response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100
)
4. Files to Migrate
✅ Already Migrated
property_hero/assets/utils.py - Refactored to use new client
📋 Pending Migration
property_hero/properties/utils.py - Uses direct OpenAI client
property_hero/asset_detection/classifier.py - Uses direct HTTP calls (async)
5. Special Cases
For Async Code
The current centralized client is synchronous. For async code (like in classifier.py), you have two options:

Option A: Create an async version of the client (recommended for async-heavy code) Option B: Use sync_to_async to call the synchronous client from async code

Example of Option B:

from asgiref.sync import sync_to_async
from property_hero.openai import get_openai_client

@sync_to_async
def classify_image(image_data):
    client = get_openai_client()
    return client.create_with_vision(
        text_prompt="Classify this image",
        images=[image_data]
    )

# In async function
result = await classify_image(base64_image)
For Vision (Image) Calls
Instead of manually building content arrays, use the built-in vision helper:

# Before
content_list = [
    {"type": "text", "text": prompt},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
]
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": content_list}]
)

# After
response = client.create_with_vision(
    text_prompt=prompt,
    images=[base64_image]
)
For JSON Responses
# Before
response = client.chat.completions.create(...)
content = response.choices[0].message.content
data = json.loads(content)

# After
data = client.chat_completion_json(messages=[...])
Testing
Mocking the Client
from unittest.mock import patch, MagicMock
from property_hero.openai import get_openai_client

def test_with_mocked_client():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"result": "test"}'

    with patch('property_hero.openai.client.get_openai_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Your test code here
        result = your_function_that_uses_openai()
        assert result == {"result": "test"}
Rollback Plan
If issues arise, you can temporarily revert by:

Keep the old import alongside the new one
Use a feature flag to switch between implementations
Monitor error rates and performance
from django.conf import settings
from property_hero.openai import get_openai_client
from openai import OpenAI as LegacyOpenAI

USE_NEW_CLIENT = getattr(settings, 'USE_NEW_OPENAI_CLIENT', True)

if USE_NEW_CLIENT:
    client = get_openai_client()
else:
    client = LegacyOpenAI(api_key=settings.OPENAI_API_KEY)
Next Steps
✅ Create centralized client module
✅ Migrate property_hero/assets/utils.py
📋 Migrate property_hero/properties/utils.py
📋 Migrate property_hero/asset_detection/classifier.py (may need async client)
📋 Update tests to use mocked client
📋 Monitor production usage and performance
Questions?
If you encounter issues during migration: 1. Check the examples in property_hero/openai/README.md 2. Review the implementation in property_hero/openai/client.py 3. Look at the migrated example in property_hero/assets/utils.py

