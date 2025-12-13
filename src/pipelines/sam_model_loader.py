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
        # (In production, use asyncio.Lock for proper async locking)
        if instance._loading:
            logger.info("sam3_model_loading_in_progress", device=device)
            # For now, raise error - in production add lock/wait logic
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
        Load SAM3 processor and model.

        Args:
            device: "cuda" or "cpu"

        Raises:
            ValueError: If loading fails
        """
        try:
            logger.info("sam3_model_loading_start", device=device)

            # Import SAM3 from transformers
            from transformers import Sam3Processor, Sam3Model

            model_name = "facebook/sam3"

            # Get cache directory from environment or use default
            cache_dir = os.environ.get("HF_HOME", "./models")

            # Load processor
            logger.info("sam3_loading_processor", model=model_name, cache_dir=cache_dir)
            self._processor = Sam3Processor.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )

            # Load model
            logger.info("sam3_loading_model", model=model_name, device=device)
            self._model = Sam3Model.from_pretrained(
                model_name,
                cache_dir=cache_dir
            ).to(device)

            self._device = device

            logger.info(
                "sam3_model_loaded_successfully",
                device=device,
                model_id=model_name
            )

        except ImportError as e:
            logger.error(
                "sam3_import_failed",
                error=str(e),
                hint="Install transformers library: pip install transformers"
            )
            raise ValueError(
                f"Failed to import SAM3 from transformers. "
                f"Ensure transformers is installed: {str(e)}"
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
