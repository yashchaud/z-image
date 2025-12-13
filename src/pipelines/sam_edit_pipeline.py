"""
SAM3 + Qwen Image Edit Pipeline.

5-step pipeline for segmentation-based image editing:
1. Segment with SAM3 using text prompt
2. Crop masked area with padding
3. Edit with Qwen Image Edit
4. Blend edited region back into original
5. Save all intermediate results
"""

import os
import time
import uuid
import torch
import numpy as np
import structlog
from PIL import Image
from typing import Dict, Any, Tuple, List, Optional

from .image_utils import poisson_blend, feather_blend, simple_paste_blend

logger = structlog.get_logger()


class SAMEditPipeline:
    """
    SAM3 API + Qwen Image Edit pipeline.

    Orchestrates a 5-step workflow for precise image editing:
    - Calls external SAM3 API for automatic segmentation from text prompts
    - Crops the region with context-preserving padding
    - Edits the cropped region with Qwen Image Edit
    - Blends the edited region seamlessly back into the original
    - Saves all intermediate results for inspection
    """

    def __init__(self, model_manager, sam_api_client, device: str, results_dir: str):
        """
        Initialize the SAM Edit pipeline.

        Args:
            model_manager: ModelManager instance (for Qwen access)
            sam_api_client: SAM3APIClient instance for segmentation
            device: "cuda" or "cpu"
            results_dir: Base directory for saving results
        """
        self.model_manager = model_manager
        self.sam_api_client = sam_api_client
        self.device = device
        self.results_dir = results_dir

    async def process(
        self,
        image: Image.Image,
        segmentation_prompt: str,
        edit_prompt: str,
        padding_percent: float = 0.25,
        blend_mode: str = "poisson",
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        response_format: str = "url"
    ) -> Dict[str, Any]:
        """
        Main orchestration method for the SAM Edit pipeline.

        Args:
            image: Input PIL Image
            segmentation_prompt: Text prompt for SAM3 (e.g., "person", "car")
            edit_prompt: Edit instruction for Qwen Image Edit
            padding_percent: Crop padding as % of bounding box (0.0-1.0)
            blend_mode: "poisson" or "feather"
            guidance_scale: Qwen guidance scale (None = use model default)
            num_inference_steps: Qwen steps (None = use model default)
            seed: Random seed for reproducibility
            response_format: "url" or "b64_json"

        Returns:
            Dictionary with final_image, intermediate_steps, and metadata

        Raises:
            ValueError: If segmentation fails or parameters invalid
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(
            "sam_edit_pipeline_start",
            request_id=request_id,
            segmentation_prompt=segmentation_prompt,
            edit_prompt=edit_prompt[:100],
            image_size=f"{image.width}x{image.height}",
            padding_percent=padding_percent,
            blend_mode=blend_mode
        )

        # Store original image
        original_image = image.copy()

        timing = {}

        try:
            # STEP 1: Segment with SAM3
            step_start = time.time()
            mask, bbox, confidence = await self._segment_with_sam3(
                image, segmentation_prompt, request_id
            )
            timing['segmentation_ms'] = int((time.time() - step_start) * 1000)

            # STEP 2: Calculate crop box and crop
            step_start = time.time()
            crop_box = self._calculate_crop_box(
                mask, bbox, (image.width, image.height), padding_percent
            )
            cropped_image, cropped_mask = self._crop_with_mask(
                image, mask, crop_box
            )
            timing['crop_ms'] = int((time.time() - step_start) * 1000)

            logger.info(
                "sam_edit_cropping",
                request_id=request_id,
                crop_box=crop_box,
                cropped_size=f"{cropped_image.width}x{cropped_image.height}"
            )

            # STEP 3: Edit with Qwen
            step_start = time.time()
            edited_image = await self._edit_with_qwen(
                cropped_image,
                cropped_mask,
                edit_prompt,
                guidance_scale,
                num_inference_steps,
                seed,
                request_id
            )
            timing['edit_ms'] = int((time.time() - step_start) * 1000)

            # STEP 4: Blend back into original
            step_start = time.time()
            final_image = self._blend_back(
                original_image,
                edited_image,
                crop_box,
                cropped_mask,
                blend_mode,
                request_id
            )
            timing['blend_ms'] = int((time.time() - step_start) * 1000)

            # STEP 5: Save intermediate results
            step_start = time.time()

            # Create mask visualization (white on black background)
            mask_vis = Image.fromarray((mask * 255).astype(np.uint8), mode='L')

            images_to_save = {
                "01_original": original_image,
                "02_mask": mask_vis,
                "03_cropped": cropped_image,
                "04_edited": edited_image,
                "05_final": final_image
            }

            folder_path, urls = self._save_intermediate_results(
                images_to_save, request_id, response_format
            )
            timing['save_ms'] = int((time.time() - step_start) * 1000)

            # Calculate total time
            total_time = time.time() - start_time
            timing['total_ms'] = int(total_time * 1000)

            logger.info(
                "sam_edit_pipeline_complete",
                request_id=request_id,
                total_seconds=round(total_time, 2),
                folder=folder_path
            )

            # Build response
            return {
                "created": int(time.time()),
                "final_image": urls["05_final"],
                "intermediate_steps": {
                    "01_original": urls["01_original"],
                    "02_mask": urls["02_mask"],
                    "03_cropped": urls["03_cropped"],
                    "04_edited": urls["04_edited"],
                    "05_final": urls["05_final"]
                },
                "metadata": {
                    "request_id": request_id,
                    "folder_path": folder_path,
                    "segmentation_prompt": segmentation_prompt,
                    "edit_prompt": edit_prompt,
                    "mask_confidence": float(confidence),
                    "crop_box": list(crop_box),
                    "padding_percent": padding_percent,
                    "blend_mode": blend_mode,
                    "timing": timing
                }
            }

        except Exception as e:
            logger.error(
                "sam_edit_pipeline_failed",
                request_id=request_id,
                error=str(e),
                step="unknown"
            )
            raise

    async def _segment_with_sam3(
        self,
        image: Image.Image,
        prompt: str,
        request_id: str
    ) -> Tuple[np.ndarray, List[int], float]:
        """
        Step 1: Segment image with SAM3 API.

        Args:
            image: PIL Image
            prompt: Text prompt for segmentation
            request_id: Request ID for logging

        Returns:
            Tuple of (binary_mask, bbox_xyxy, confidence_score)

        Raises:
            ValueError: If no objects found matching prompt
        """
        try:
            logger.info("sam3_api_segmentation_start", request_id=request_id, prompt=prompt)

            # Call external SAM3 API
            result = await self.sam_api_client.segment(
                image=image,
                text_prompt=prompt,
                threshold=0.5,
                mask_threshold=0.5
            )

            # Extract results from API response
            masks_b64 = result["masks"]  # List of base64-encoded mask PNGs
            boxes = result["boxes"]  # List of [x1, y1, x2, y2]
            scores = result["scores"]  # List of confidence scores
            count = result["count"]

            # Check if any objects found
            if count == 0:
                raise ValueError(
                    f"No objects found matching segmentation prompt: '{prompt}'. "
                    f"Please try a different prompt or verify the object exists in the image."
                )

            # Use highest confidence mask
            best_idx = int(np.argmax(scores))

            # Decode mask from base64
            mask_image = self.sam_api_client.decode_mask(masks_b64[best_idx])
            mask = np.array(mask_image) > 0  # Convert to boolean mask

            bbox = boxes[best_idx]  # [x1, y1, x2, y2]
            confidence = float(scores[best_idx])

            # Log selection if multiple objects found
            if count > 1:
                logger.info(
                    "sam3_multiple_objects_found",
                    request_id=request_id,
                    num_objects=count,
                    selected_index=best_idx,
                    selected_confidence=confidence,
                    all_confidences=scores
                )

            # Warn if low confidence
            if confidence < 0.3:
                logger.warning(
                    "sam3_low_confidence",
                    request_id=request_id,
                    confidence=confidence,
                    prompt=prompt
                )

            logger.info(
                "sam3_api_segmentation_complete",
                request_id=request_id,
                confidence=confidence,
                bbox=bbox
            )

            return mask, bbox, confidence

        except Exception as e:
            logger.error(
                "sam3_api_segmentation_failed",
                request_id=request_id,
                error=str(e)
            )
            raise

    def _calculate_crop_box(
        self,
        mask: np.ndarray,
        bbox: List[int],
        image_size: Tuple[int, int],
        padding_percent: float
    ) -> Tuple[int, int, int, int]:
        """
        Step 2a: Calculate crop box with padding.

        Args:
            mask: Binary mask from SAM3
            bbox: [x, y, w, h] bounding box from SAM3
            image_size: (width, height) of original image
            padding_percent: Padding as % of bbox size (0.0-1.0)

        Returns:
            (x1, y1, x2, y2) crop box clamped to image boundaries
        """
        image_width, image_height = image_size
        x, y, w, h = bbox

        # Calculate padding in pixels
        pad_x = int(w * padding_percent)
        pad_y = int(h * padding_percent)

        # Apply padding and clamp to boundaries
        crop_x1 = max(0, int(x - pad_x))
        crop_y1 = max(0, int(y - pad_y))
        crop_x2 = min(image_width, int(x + w + pad_x))
        crop_y2 = min(image_height, int(y + h + pad_y))

        # Validate minimum crop size
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1

        MIN_CROP_SIZE = 64
        if crop_width < MIN_CROP_SIZE or crop_height < MIN_CROP_SIZE:
            raise ValueError(
                f"Crop region too small ({crop_width}x{crop_height}). "
                f"Minimum size is {MIN_CROP_SIZE}x{MIN_CROP_SIZE}. "
                f"Try reducing padding_percent."
            )

        return (crop_x1, crop_y1, crop_x2, crop_y2)

    def _crop_with_mask(
        self,
        image: Image.Image,
        mask: np.ndarray,
        crop_box: Tuple[int, int, int, int]
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Step 2b: Crop image and mask.

        Args:
            image: PIL Image
            mask: Binary mask as numpy array
            crop_box: (x1, y1, x2, y2)

        Returns:
            (cropped_image, cropped_mask)
        """
        x1, y1, x2, y2 = crop_box

        # Crop image
        cropped_img = image.crop((x1, y1, x2, y2))

        # Convert mask to PIL and crop
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        cropped_mask = mask_pil.crop((x1, y1, x2, y2))

        return cropped_img, cropped_mask

    async def _edit_with_qwen(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        guidance_scale: Optional[float],
        num_steps: Optional[int],
        seed: Optional[int],
        request_id: str
    ) -> Image.Image:
        """
        Step 3: Edit cropped image with Qwen Image Edit.

        Args:
            image: Cropped image
            mask: Cropped mask
            prompt: Edit instruction
            guidance_scale: Guidance scale (None = use default)
            num_steps: Inference steps (None = use default)
            seed: Random seed
            request_id: Request ID for logging

        Returns:
            Edited PIL Image

        Raises:
            ValueError: If Qwen pipeline fails
        """
        try:
            logger.info("qwen_edit_start", request_id=request_id, prompt=prompt[:100])

            # Get Qwen inpainting pipeline
            pipeline = self.model_manager.inpaint_pipeline

            if pipeline is None:
                raise ValueError("Qwen inpainting pipeline not available")

            # Use model defaults if not specified
            if guidance_scale is None:
                guidance_scale = 7.5
            if num_steps is None:
                num_steps = 30

            # Set up generator for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)

            # Run Qwen edit
            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    image=image,
                    mask_image=mask,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    generator=generator
                )

            edited_img = result.images[0]

            logger.info(
                "qwen_edit_complete",
                request_id=request_id,
                output_size=f"{edited_img.width}x{edited_img.height}"
            )

            return edited_img

        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                "qwen_oom",
                request_id=request_id,
                crop_size=f"{image.width}x{image.height}"
            )
            raise ValueError(
                f"Out of memory during image editing. "
                f"Try using a smaller padding_percent to reduce crop size."
            )

        except Exception as e:
            logger.error(
                "qwen_edit_failed",
                request_id=request_id,
                error=str(e)
            )
            raise

    def _blend_back(
        self,
        original: Image.Image,
        edited: Image.Image,
        crop_box: Tuple[int, int, int, int],
        cropped_mask: Image.Image,
        blend_mode: str,
        request_id: str
    ) -> Image.Image:
        """
        Step 4: Blend edited region back into original.

        Args:
            original: Original full image
            edited: Edited cropped region
            crop_box: (x1, y1, x2, y2) of crop in original
            cropped_mask: Mask for cropped region
            blend_mode: "poisson" or "feather"
            request_id: Request ID for logging

        Returns:
            Final blended image
        """
        try:
            logger.info("blending_start", request_id=request_id, mode=blend_mode)

            x1, y1, x2, y2 = crop_box

            # Resize edited image to match crop size if needed
            crop_width = x2 - x1
            crop_height = y2 - y1

            if edited.width != crop_width or edited.height != crop_height:
                edited = edited.resize((crop_width, crop_height), Image.Resampling.LANCZOS)

            # Create full-size canvas
            result = original.copy()

            if blend_mode == "poisson":
                # Poisson blending requires full-size images
                # Create full-size edited canvas
                full_edited = original.copy()
                full_edited.paste(edited, (x1, y1))

                # Create full-size mask
                full_mask = Image.new('L', original.size, 0)
                full_mask.paste(cropped_mask, (x1, y1))

                # Calculate center point
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Try Poisson blend, fallback to feather if it fails
                try:
                    result = poisson_blend(full_edited, original, full_mask, center)
                    logger.info("poisson_blend_success", request_id=request_id)

                except Exception as e:
                    logger.warning(
                        "poisson_blend_failed_fallback_to_feather",
                        request_id=request_id,
                        error=str(e)
                    )
                    # Fallback to feather blend
                    result = feather_blend(full_edited, original, full_mask, feather_radius=10)

            elif blend_mode == "feather":
                # Feather blending
                full_edited = original.copy()
                full_edited.paste(edited, (x1, y1))

                full_mask = Image.new('L', original.size, 0)
                full_mask.paste(cropped_mask, (x1, y1))

                result = feather_blend(full_edited, original, full_mask, feather_radius=10)
                logger.info("feather_blend_success", request_id=request_id)

            else:
                logger.warning(
                    "unknown_blend_mode_using_simple_paste",
                    request_id=request_id,
                    mode=blend_mode
                )
                # Simple paste as fallback
                result.paste(edited, (x1, y1), cropped_mask)

            logger.info("blending_complete", request_id=request_id, mode=blend_mode)

            return result

        except Exception as e:
            logger.error(
                "blending_failed",
                request_id=request_id,
                error=str(e)
            )
            # Last resort: simple paste
            logger.warning("using_simple_paste_fallback", request_id=request_id)
            result = original.copy()
            result.paste(edited, (x1, y1), cropped_mask)
            return result

    def _save_intermediate_results(
        self,
        images: Dict[str, Image.Image],
        request_id: str,
        response_format: str
    ) -> Tuple[str, Dict[str, Dict]]:
        """
        Step 5: Save all intermediate results.

        Creates a unique folder and saves all step images as PNGs.

        Args:
            images: Dict mapping step names to PIL Images
            request_id: Request ID for logging
            response_format: "url" or "b64_json"

        Returns:
            Tuple of (folder_path, url_dict)
        """
        try:
            # Create unique folder
            timestamp = int(time.time() * 1000)
            unique_id = str(uuid.uuid4())[:8]
            folder_name = f"{timestamp}_{unique_id}"

            folder_path = os.path.join(self.results_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            logger.info(
                "saving_results",
                request_id=request_id,
                folder=folder_path,
                num_images=len(images)
            )

            # Save all images
            urls = {}
            for step_name, img in images.items():
                filename = f"{step_name}.png"
                filepath = os.path.join(folder_path, filename)
                img.save(filepath, format="PNG")

                # Build URL or base64 response
                if response_format == "url":
                    urls[step_name] = {
                        "url": f"/results/{folder_name}/{filename}"
                    }
                else:  # b64_json
                    from io import BytesIO
                    import base64

                    buffer = BytesIO()
                    img.save(buffer, format="PNG")
                    b64_string = base64.b64encode(buffer.getvalue()).decode()

                    urls[step_name] = {
                        "b64_json": b64_string
                    }

            logger.info(
                "results_saved",
                request_id=request_id,
                folder=folder_path
            )

            return folder_path, urls

        except Exception as e:
            logger.error(
                "save_results_failed",
                request_id=request_id,
                error=str(e)
            )
            raise
