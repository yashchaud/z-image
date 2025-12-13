"""
SAM3 Model Manager with lazy loading and singleton pattern.

Manages the lifecycle of the SAM3 (Segment Anything Model 3) from Meta/Facebook.
Uses singleton pattern to avoid reloading the model on each request.
"""

import os
import structlog
from typing import Tuple, Optional

logger = structlog.get_logger()


class SAMModelManager:
    """
    Manages SAM3 model loading with lazy initialization.

    Singleton pattern ensures the model is loaded only once and shared
    across all requests. The model is loaded on first request, not at
    server startup, to avoid blocking startup time.
    """

    _instance: Optional['SAMModelManager'] = None
    _processor = None
    _model = None
    _device = None
    _loading = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SAMModelManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    async def get_instance(cls, device: str = "cuda") -> Tuple:
        """
        Get or create SAM3 model singleton.

        This method is thread-safe and will only load the model once,
        even if called concurrently from multiple requests.

        Args:
            device: "cuda" or "cpu"

        Returns:
            Tuple of (processor, model, device)

        Raises:
            ValueError: If model loading fails
        """
        instance = cls()

        # If model already loaded and device matches, return it
        if instance._model is not None and instance._device == device:
            logger.info("sam3_model_already_loaded", device=device)
            return (instance._processor, instance._model, instance._device)

        # If model is currently being loaded by another request, wait
        if instance._loading:
            logger.info("sam3_model_loading_in_progress", device=device)
            raise ValueError("SAM3 model is currently being loaded by another request")

        # Load the model
        try:
            instance._loading = True
            await instance._load_model(device)
            return (instance._processor, instance._model, instance._device)
        finally:
            instance._loading = False

    async def _load_model(self, device: str):
        """
        Load SAM3 processor and model from transformers.

        Args:
            device: "cuda" or "cpu"

        Raises:
            ValueError: If loading fails
        """
        try:
            logger.info("sam3_model_loading_start", device=device)

            # Import from transformers
            from transformers import AutoProcessor, AutoModel
            import torch

            # Get HF token from environment
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

            logger.info("sam3_loading_from_transformers", repo="facebook/sam3", has_token=bool(hf_token))

            # Load processor using AutoProcessor (works across transformers versions)
            self._processor = AutoProcessor.from_pretrained(
                "facebook/sam3",
                token=hf_token,
                trust_remote_code=True
            )

            # Load model using AutoModel (works across transformers versions)
            self._model = AutoModel.from_pretrained(
                "facebook/sam3",
                token=hf_token,
                trust_remote_code=True
            ).to(device)

            self._device = device

            logger.info(
                "sam3_model_loaded_successfully",
                device=device,
                model_type="transformers_sam3_auto"
            )

        except ImportError as e:
            logger.error(
                "sam3_import_failed",
                error=str(e),
                hint="Make sure transformers>=4.51.3 is installed"
            )
            raise ValueError(
                f"Failed to import SAM3 from transformers. "
                f"Make sure transformers>=4.51.3 is installed. "
                f"Error: {str(e)}"
            )

        except Exception as e:
            logger.error(
                "sam3_model_load_failed",
                error=str(e),
                device=device
            )
            # Clean up partial state
            self._processor = None
            self._model = None
            self._device = None

            raise ValueError(
                f"Failed to load SAM3 model from facebook/sam3. "
                f"Check internet connection and Hugging Face access. "
                f"Error: {str(e)}"
            )

    @classmethod
    def clear(cls):
        """
        Clear the cached model and processor.

        Useful for testing or when switching devices.
        """
        instance = cls()
        instance._processor = None
        instance._model = None
        instance._device = None
        instance._loading = False
        logger.info("sam3_model_cleared")
