import json
import os
from . import config


class AssetClassifier:
    """Handles OpenAI-based asset classification and value estimation"""

    def __init__(self):
        # Get API key from environment variable only
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")

    async def classify_async(self, session, base64_image, index):
        """Send image to OpenAI for classification and value estimation asynchronously"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }

        payload = {
            "model": config.OPENAI_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Identify this object and estimate its replacement value. Respond with JSON only:
{
  "name": "object name (1-2 words)",
  "estimated_value": replacement value in USD (number only)
}

Example: {"name": "Sofa", "estimated_value": 700}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 100,
            "response_format": {"type": "json_object"}
        }

        try:
            async with session.post("https://api.openai.com/v1/chat/completions",
                                  headers=headers, json=payload, timeout=30) as response:
                response.raise_for_status()
                result = await response.json()
                content = result['choices'][0]['message']['content'].strip()

                # Parse JSON response
                data = json.loads(content)
                classification = {
                    'name': data.get('name', 'Unknown Item'),
                    'estimated_value': int(data.get('estimated_value', 100))
                }

                print(f"✓ OpenAI classified object {index}: {classification['name']} (${classification['estimated_value']} ± 50)")
                return classification

        except json.JSONDecodeError as e:
            print(f"OpenAI JSON parse error for object {index}: {e}")
            return {'name': 'Unknown Item', 'estimated_value': 100}
        except Exception as e:
            print(f"OpenAI API error for object {index}: {e}")
            return {'name': 'Unknown Item', 'estimated_value': 100}
