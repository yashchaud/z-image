"""
OpenRouter API client for generating creative LinkedIn post prompts.
"""

import asyncio
import json
from typing import List
import httpx
import structlog

logger = structlog.get_logger()

LINKEDIN_SYSTEM_PROMPT = """You are a creative visual designer specializing in LinkedIn content.

Your task: Generate exactly 2 distinct, visually striking image prompts for a LinkedIn post based on:
1. The user's post context/idea
2. A reference image they provide

Requirements for each prompt:
- Professional yet eye-catching for LinkedIn feed
- Maintain brand consistency with reference image (colors, style, tone)
- Create visual variety between the 2 prompts (different compositions, angles, or emphasis)
- Keep prompts concise (50-100 words each)
- Focus on visual elements, not text overlays
- Suitable for business/professional context

Output format (strict JSON):
{
  "prompts": [
    "First detailed image prompt here...",
    "Second detailed image prompt here..."
  ]
}

Examples of good prompts:
- "Modern minimalist office setting, bright natural lighting from floor-to-ceiling windows, laptop on clean white desk with coffee cup, shallow depth of field, professional business atmosphere, 4K quality, corporate photography style"
- "Dynamic team collaboration scene, diverse professionals around conference table with laptops and charts, warm ambient lighting, slight motion blur on gesturing hands, cinematic composition, inspirational business mood"

Ensure the prompts are:
1. Specific about composition, lighting, and mood
2. Aligned with the reference image's aesthetic
3. Professionally appropriate for LinkedIn
4. Visually distinct from each other"""


class OpenRouterClient:
    """
    Client for OpenRouter API to generate creative prompts using vision models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen/qwen-2-vl-72b-instruct",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            model: Model identifier (default: qwen vision model)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = httpx.AsyncClient(timeout=timeout)

    async def generate_creative_prompts(
        self,
        text: str,
        reference_image_b64: str,
        request_id: str
    ) -> List[str]:
        """
        Generate 2 creative LinkedIn prompts using vision model.

        Args:
            text: User's post idea or context
            reference_image_b64: Base64 encoded reference image
            request_id: Request ID for logging

        Returns:
            List of 2 creative prompts

        Raises:
            ValueError: If response is invalid or prompts are missing
            httpx.HTTPError: If API call fails after retries
        """
        logger.info("openrouter_request_start",
                   request_id=request_id,
                   model=self.model,
                   text_length=len(text))

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": LINKEDIN_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Context: {text}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{reference_image_b64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.8,
            "max_tokens": 800,
            "response_format": {"type": "json_object"}
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://z-image-linkedin",
            "X-Title": "Z-Image LinkedIn Pipeline"
        }

        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                logger.info("openrouter_api_call",
                           request_id=request_id,
                           attempt=attempt + 1,
                           max_retries=self.max_retries)

                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                )

                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning("openrouter_rate_limited",
                                 request_id=request_id,
                                 retry_after=retry_after,
                                 attempt=attempt + 1)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(retry_after)
                        continue
                    else:
                        raise httpx.HTTPStatusError(
                            f"Rate limited after {self.max_retries} attempts",
                            request=response.request,
                            response=response
                        )

                # Raise for other HTTP errors
                response.raise_for_status()

                # Parse response
                response_data = response.json()

                # Extract content from response
                if "choices" not in response_data or len(response_data["choices"]) == 0:
                    raise ValueError("No choices in OpenRouter response")

                content = response_data["choices"][0]["message"]["content"]

                # Parse JSON from content
                try:
                    prompts_data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error("openrouter_json_parse_failed",
                               request_id=request_id,
                               content=content[:500],
                               error=str(e))
                    raise ValueError(f"Failed to parse JSON from OpenRouter: {str(e)}")

                # Validate prompts
                if "prompts" not in prompts_data:
                    raise ValueError("No 'prompts' key in OpenRouter response")

                prompts = prompts_data["prompts"]

                if not isinstance(prompts, list) or len(prompts) != 2:
                    raise ValueError(f"Expected 2 prompts, got {len(prompts) if isinstance(prompts, list) else 'invalid type'}")

                for i, prompt in enumerate(prompts):
                    if not isinstance(prompt, str) or len(prompt) < 20:
                        raise ValueError(f"Prompt {i} is invalid or too short")

                logger.info("openrouter_prompts_generated",
                           request_id=request_id,
                           prompt1_len=len(prompts[0]),
                           prompt2_len=len(prompts[1]),
                           attempt=attempt + 1)

                return prompts

            except httpx.TimeoutException as e:
                logger.warning("openrouter_timeout",
                             request_id=request_id,
                             attempt=attempt + 1,
                             error=str(e))
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error("openrouter_timeout_final",
                               request_id=request_id,
                               max_retries=self.max_retries)
                    raise ValueError(f"OpenRouter timed out after {self.max_retries} attempts")

            except httpx.HTTPStatusError as e:
                logger.error("openrouter_http_error",
                           request_id=request_id,
                           status_code=e.response.status_code,
                           error=str(e),
                           response_text=e.response.text[:500])

                # Don't retry for 4xx errors (except 429 handled above)
                if 400 <= e.response.status_code < 500:
                    raise ValueError(f"OpenRouter API error: {e.response.text[:200]}")

                # Retry for 5xx errors
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise

            except Exception as e:
                logger.error("openrouter_unexpected_error",
                           request_id=request_id,
                           attempt=attempt + 1,
                           error=str(e))

                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise ValueError(f"OpenRouter failed: {str(e)}")

        # Should never reach here, but just in case
        raise ValueError(f"OpenRouter failed after {self.max_retries} attempts")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
