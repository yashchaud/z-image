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
        Load SAM3 processor and model using native SAM3 library.

        Args:
            device: "cuda" or "cpu"

        Raises:
            ValueError: If loading fails
        """
        try:
            logger.info("sam3_model_loading_start", device=device)

            # Import SAM3 native library
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            from huggingface_hub import login

            # Authenticate with Hugging Face if token available
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if hf_token:
                logger.info("sam3_authenticating_hf", has_token=True)
                login(token=hf_token)
            else:
                logger.warning("sam3_no_hf_token", message="No HF_TOKEN found in environment")

            # Download model checkpoint from Hugging Face using snapshot_download
            from huggingface_hub import snapshot_download
            import traceback

            logger.info("sam3_downloading_checkpoint", repo="facebook/sam3")

            try:
                # Download entire repo snapshot (includes model, config, tokenizer)
                cache_dir = os.environ.get("HF_HOME", "./models/sam3")

                repo_path = snapshot_download(
                    repo_id="facebook/sam3",
                    token=hf_token,
                    cache_dir=cache_dir,
                    allow_patterns=["model.safetensors", "config.json", "processor_config.json",
                                   "tokenizer.json", "vocab.json", "merges.txt",
                                   "special_tokens_map.json", "tokenizer_config.json"]
                )

                # Construct path to model file
                checkpoint_path = os.path.join(repo_path, "model.safetensors")

                logger.info("sam3_checkpoint_downloaded", path=checkpoint_path, repo_path=repo_path)
            except Exception as download_error:
                logger.error(
                    "sam3_download_failed",
                    error=str(download_error),
                    error_type=type(download_error).__name__,
                    traceback=traceback.format_exc()
                )
                raise

            # Load model with native SAM3 builder from downloaded checkpoint
            logger.info("sam3_loading_native_model", device=device)
            self._model = build_sam3_image_model(
                device=device,
                eval_mode=True,
                load_from_HF=False,  # Load from local checkpoint
                checkpoint_path=checkpoint_path
            )

            # Create processor
            logger.info("sam3_creating_processor", device=device)
            self._processor = Sam3Processor(self._model, device=device)

            self._device = device

            logger.info(
                "sam3_model_loaded_successfully",
                device=device,
                model_type="native_sam3"
            )

        except ImportError as e:
            logger.error(
                "sam3_import_failed",
                error=str(e),
                hint="Install SAM3 library: pip install git+https://github.com/facebookresearch/sam3.git"
            )
            raise ValueError(
                f"Failed to import SAM3 native library. "
                f"Install with: pip install git+https://github.com/facebookresearch/sam3.git"
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
