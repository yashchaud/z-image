"""
SAM3 API Client for external segmentation service.

Makes HTTP calls to a separate SAM3 microservice for text-based segmentation.
"""

import os
import base64
import structlog
from typing import Tuple, Dict, Any
from io import BytesIO
from PIL import Image
import httpx

logger = structlog.get_logger()


class SAM3APIClient:
    """
    Client for calling external SAM3 segmentation API.

    Makes HTTP requests to a SAM3 microservice instead of loading
    the model locally. This avoids dependency conflicts and allows
    independent scaling.
    """

    def __init__(self, api_url: str = None, timeout: float = 120.0):
        """
        Initialize SAM3 API client.

        Args:
            api_url: Base URL of SAM3 service (e.g., "http://sam3-service:8001")
                    Falls back to SAM3_API_URL environment variable
            timeout: Request timeout in seconds (default: 120s for model loading)
        """
        self.api_url = api_url or os.environ.get("SAM3_API_URL")
        if not self.api_url:
            raise ValueError(
                "SAM3_API_URL must be set in environment or passed to constructor. "
                "Example: SAM3_API_URL=http://localhost:8001"
            )

        self.timeout = timeout
        self.segment_endpoint = f"{self.api_url.rstrip('/')}/v1/segment"

        logger.info(
            "sam3_api_client_initialized",
            api_url=self.api_url,
            endpoint=self.segment_endpoint,
            timeout=timeout
        )

    async def segment(
        self,
        image: Image.Image,
        text_prompt: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Call SAM3 API to segment image with text prompt.

        Args:
            image: PIL Image to segment
            text_prompt: Text description of object to segment (e.g., "person", "car")
            threshold: Detection confidence threshold (0.0-1.0)
            mask_threshold: Mask confidence threshold (0.0-1.0)

        Returns:
            Dictionary containing:
                - masks: List of binary masks as base64-encoded PNG strings
                - boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
                - scores: List of confidence scores
                - count: Number of detected objects

        Raises:
            httpx.HTTPError: If API request fails
            ValueError: If API returns invalid response
        """
        try:
            logger.info(
                "sam3_api_request_start",
                endpoint=self.segment_endpoint,
                text_prompt=text_prompt,
                threshold=threshold
            )

            # Convert image to base64
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Prepare request payload
            payload = {
                "image": image_b64,
                "text_prompt": text_prompt,
                "threshold": threshold,
                "mask_threshold": mask_threshold
            }

            # Make async HTTP request
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.segment_endpoint,
                    json=payload
                )
                response.raise_for_status()

            result = response.json()

            logger.info(
                "sam3_api_request_success",
                num_objects=result.get("count", 0),
                endpoint=self.segment_endpoint
            )

            return result

        except httpx.HTTPStatusError as e:
            logger.error(
                "sam3_api_http_error",
                status_code=e.response.status_code,
                error=str(e),
                endpoint=self.segment_endpoint
            )
            raise ValueError(
                f"SAM3 API returned error {e.response.status_code}: {e.response.text}"
            )

        except httpx.RequestError as e:
            logger.error(
                "sam3_api_connection_error",
                error=str(e),
                endpoint=self.segment_endpoint
            )
            raise ValueError(
                f"Failed to connect to SAM3 API at {self.api_url}. "
                f"Ensure SAM3 service is running. Error: {str(e)}"
            )

        except Exception as e:
            logger.error(
                "sam3_api_unexpected_error",
                error=str(e),
                endpoint=self.segment_endpoint
            )
            raise

    @staticmethod
    def decode_mask(mask_b64: str) -> Image.Image:
        """
        Decode base64-encoded mask PNG to PIL Image.

        Args:
            mask_b64: Base64-encoded PNG string

        Returns:
            PIL Image (grayscale mask)
        """
        mask_bytes = base64.b64decode(mask_b64)
        mask_image = Image.open(BytesIO(mask_bytes))
        return mask_image
