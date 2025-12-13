"""
GroundingDINO + SAM2 Model Manager with lazy loading and singleton pattern.

Manages the lifecycle of GroundingDINO (text-based detection) and SAM2 (segmentation).
Uses singleton pattern to avoid reloading the models on each request.
"""

import os
import torch
import structlog
from typing import Tuple, Optional

logger = structlog.get_logger()


class SAMModelManager:
    """
    Manages GroundingDINO + SAM2 model loading with lazy initialization.

    Singleton pattern ensures the models are loaded only once and shared
    across all requests. The models are loaded on first request, not at
    server startup, to avoid blocking startup time.
    """

    _instance: Optional['SAMModelManager'] = None
    _grounding_model = None
    _sam_predictor = None
    _device = None
    _loading = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SAMModelManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    async def get_instance(cls, device: str = "cuda") -> Tuple:
        """
        Get or create GroundingDINO + SAM2 model singleton.

        This method is thread-safe and will only load the models once,
        even if called concurrently from multiple requests.

        Args:
            device: "cuda" or "cpu"

        Returns:
            Tuple of (grounding_model, sam_predictor, device)

        Raises:
            ValueError: If model loading fails
        """
        instance = cls()

        # If models already loaded and device matches, return them
        if instance._grounding_model is not None and instance._device == device:
            logger.info("segmentation_models_already_loaded", device=device)
            return (instance._grounding_model, instance._sam_predictor, instance._device)

        # If models are currently being loaded by another request, wait
        if instance._loading:
            logger.info("segmentation_models_loading_in_progress", device=device)
            raise ValueError("Segmentation models are currently being loaded by another request")

        # Load the models
        try:
            instance._loading = True
            await instance._load_model(device)
            return (instance._grounding_model, instance._sam_predictor, instance._device)
        finally:
            instance._loading = False

    async def _load_model(self, device: str):
        """
        Load GroundingDINO and SAM2 models.

        Args:
            device: "cuda" or "cpu"

        Raises:
            ValueError: If loading fails
        """
        try:
            logger.info("segmentation_models_loading_start", device=device)

            # Import required libraries
            from groundingdino.util.inference import load_model as load_grounding_model
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from huggingface_hub import hf_hub_download
            import traceback

            # 1. Load GroundingDINO
            logger.info("grounding_dino_loading", device=device)
            try:
                # Download GroundingDINO config and checkpoint
                grounding_config_path = hf_hub_download(
                    repo_id="ShilongLiu/GroundingDINO",
                    filename="GroundingDINO_SwinT_OGC.cfg.py"
                )
                grounding_checkpoint_path = hf_hub_download(
                    repo_id="ShilongLiu/GroundingDINO",
                    filename="groundingdino_swint_ogc.pth"
                )

                self._grounding_model = load_grounding_model(
                    grounding_config_path,
                    grounding_checkpoint_path,
                    device=device
                )
                logger.info("grounding_dino_loaded_successfully", device=device)
            except Exception as grounding_error:
                logger.error(
                    "grounding_dino_load_failed",
                    error=str(grounding_error),
                    traceback=traceback.format_exc()
                )
                raise

            # 2. Load SAM2
            logger.info("sam2_loading", device=device)
            try:
                # Download SAM2 checkpoint
                sam2_checkpoint_path = hf_hub_download(
                    repo_id="facebook/sam2-hiera-large",
                    filename="sam2_hiera_large.pt"
                )

                # Build SAM2 model
                sam2_model = build_sam2(
                    config_file="sam2_hiera_l.yaml",
                    ckpt_path=sam2_checkpoint_path,
                    device=device
                )

                # Create predictor
                self._sam_predictor = SAM2ImagePredictor(sam2_model)
                logger.info("sam2_loaded_successfully", device=device)
            except Exception as sam2_error:
                logger.error(
                    "sam2_load_failed",
                    error=str(sam2_error),
                    traceback=traceback.format_exc()
                )
                raise

            self._device = device

            logger.info(
                "segmentation_models_loaded_successfully",
                device=device,
                models="GroundingDINO + SAM2"
            )

        except ImportError as e:
            logger.error(
                "segmentation_import_failed",
                error=str(e),
                hint="Install required libraries: pip install groundingdino-py segment-anything-2"
            )
            raise ValueError(
                f"Failed to import segmentation libraries. "
                f"Install with: pip install groundingdino-py segment-anything-2 "
                f"Error: {str(e)}"
            )

        except Exception as e:
            logger.error(
                "segmentation_model_load_failed",
                error=str(e),
                device=device
            )
            # Clean up partial state
            self._grounding_model = None
            self._sam_predictor = None
            self._device = None

            raise ValueError(
                f"Failed to load segmentation models. "
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
