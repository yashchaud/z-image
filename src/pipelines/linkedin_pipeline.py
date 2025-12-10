"""
LinkedIn post image generation pipeline.

This module orchestrates the end-to-end process of generating
professional LinkedIn post images with creative variations.
"""

import asyncio
import time
import uuid
from typing import Optional, List, Dict, Any
import torch
import structlog

from .openrouter_client import OpenRouterClient

logger = structlog.get_logger()


class LinkedInPipeline:
    """
    LinkedIn post image generation pipeline.

    Flow:
    1. Validate reference image
    2. Call OpenRouter to generate 2 creative prompts
    3. Generate 2 images in parallel using z-image
    4. Return variants with metadata
    """

    def __init__(self, model_manager, openrouter_client: OpenRouterClient):
        """
        Initialize LinkedIn pipeline.

        Args:
            model_manager: ModelManager instance from server.py
            openrouter_client: OpenRouterClient instance
        """
        self.model_manager = model_manager
        self.openrouter_client = openrouter_client
        self.logger = structlog.get_logger()

    async def process(
        self,
        text: str,
        reference_image_b64: str,
        size: str = "1024x1024",
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        response_format: str = "url"
    ) -> Dict[str, Any]:
        """
        Main orchestration method for LinkedIn post generation.

        Args:
            text: User's post idea or context
            reference_image_b64: Base64 encoded reference image
            size: Image size in WxH format
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            seed: Random seed for reproducibility
            response_format: "url" or "b64_json"

        Returns:
            Dictionary with variants and metadata
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        self.logger.info("linkedin_pipeline_start",
                        request_id=request_id,
                        text_length=len(text),
                        size=size)

        # Step 1: Generate creative prompts from OpenRouter
        prompt_start = time.time()
        try:
            prompts = await self.openrouter_client.generate_creative_prompts(
                text=text,
                reference_image_b64=reference_image_b64,
                request_id=request_id
            )
            prompt_time = time.time() - prompt_start

            self.logger.info("creative_prompts_received",
                           request_id=request_id,
                           num_prompts=len(prompts),
                           elapsed=round(prompt_time, 2))

        except Exception as e:
            self.logger.error("prompt_generation_failed",
                            request_id=request_id,
                            error=str(e))
            raise

        # Step 2: Generate images in parallel
        img_start = time.time()
        try:
            variants = await self._generate_images_parallel(
                prompts=prompts,
                size=size,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                response_format=response_format,
                request_id=request_id
            )
            img_time = time.time() - img_start

            self.logger.info("images_generated",
                           request_id=request_id,
                           num_variants=len(variants),
                           elapsed=round(img_time, 2))

        except Exception as e:
            self.logger.error("image_generation_failed",
                            request_id=request_id,
                            error=str(e))
            raise

        total_time = time.time() - start_time

        return {
            "created": int(time.time()),
            "variants": variants,
            "metadata": {
                "openrouter_model": self.openrouter_client.model,
                "prompt_generation_time": round(prompt_time, 2),
                "image_generation_time": round(img_time, 2),
                "total_time": round(total_time, 2),
                "reference_image_size": size,
                "request_id": request_id
            }
        }

    async def _generate_images_parallel(
        self,
        prompts: List[str],
        size: str,
        guidance_scale: Optional[float],
        num_inference_steps: Optional[int],
        seed: Optional[int],
        response_format: str,
        request_id: str
    ) -> List[Dict[str, Any]]:
        """
        Generate images in parallel using asyncio.gather.

        Args:
            prompts: List of creative prompts from OpenRouter
            size: Image size in WxH format
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            seed: Random seed base (will increment for each variant)
            response_format: "url" or "b64_json"
            request_id: Request ID for logging

        Returns:
            List of variant dictionaries with urls/b64, prompts, seeds
        """
        # Parse size
        width, height = map(int, size.split("x"))

        # Get default parameters
        default_params = self.model_manager.config.get("default_params", {})
        guidance_scale = guidance_scale or default_params.get("guidance_scale", 7.5)
        num_steps = num_inference_steps or default_params.get("num_inference_steps", 30)

        self.logger.info("preparing_parallel_generation",
                        request_id=request_id,
                        num_prompts=len(prompts),
                        width=width,
                        height=height,
                        guidance_scale=guidance_scale,
                        num_steps=num_steps)

        # Create tasks for parallel execution
        tasks = []
        for i, prompt in enumerate(prompts):
            # Use different seeds for each variant if seed provided
            variant_seed = (seed + i) if seed is not None else None
            task = self._generate_single_image(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                seed=variant_seed,
                response_format=response_format,
                request_id=request_id,
                variant_index=i
            )
            tasks.append(task)

        # Execute in parallel
        self.logger.info("parallel_generation_start",
                        request_id=request_id,
                        count=len(tasks))

        variants = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle partial failures
        successful_variants = []
        failed_indices = []

        for i, variant in enumerate(variants):
            if isinstance(variant, Exception):
                self.logger.error("variant_generation_failed",
                                request_id=request_id,
                                index=i,
                                error=str(variant))
                failed_indices.append(i)
            else:
                successful_variants.append(variant)

        # Strict mode: Require all variants to succeed
        if failed_indices:
            raise ValueError(f"Generation failed for variants: {failed_indices}")

        self.logger.info("parallel_generation_complete",
                        request_id=request_id,
                        successful=len(successful_variants),
                        failed=len(failed_indices))

        return successful_variants

    async def _generate_single_image(
        self,
        prompt: str,
        width: int,
        height: int,
        guidance_scale: float,
        num_steps: int,
        seed: Optional[int],
        response_format: str,
        request_id: str,
        variant_index: int
    ) -> Dict[str, Any]:
        """
        Generate a single image using the text2img pipeline.

        Args:
            prompt: Creative prompt for generation
            width: Image width
            height: Image height
            guidance_scale: Guidance scale
            num_steps: Number of inference steps
            seed: Random seed
            response_format: "url" or "b64_json"
            request_id: Request ID for logging
            variant_index: Index of this variant (0 or 1)

        Returns:
            Dictionary with url/b64_json, prompt, and seed
        """
        self.logger.info("generating_variant",
                        request_id=request_id,
                        index=variant_index,
                        prompt=prompt[:100],
                        seed=seed)

        # Import utilities from server module
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from server import save_image_to_assets, encode_image_to_base64

        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.model_manager.device).manual_seed(seed)

        variant_start = time.time()

        # Acquire semaphore and generate
        # Note: ModelManager should have _semaphore instead of _lock for parallel execution
        async with self.model_manager._semaphore:
            with torch.inference_mode():
                result = self.model_manager.text2img_pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    generator=generator
                )
                image = result.images[0]

        variant_time = time.time() - variant_start

        self.logger.info("variant_generated",
                        request_id=request_id,
                        index=variant_index,
                        elapsed=round(variant_time, 2))

        # Encode response
        if response_format == "url":
            filename = save_image_to_assets(image, prefix=f"linkedin_v{variant_index}")
            url = f"/assets/{filename}"
            return {
                "url": url,
                "prompt": prompt,
                "seed": seed if seed is not None else -1
            }
        else:
            b64 = encode_image_to_base64(image)
            return {
                "b64_json": b64,
                "prompt": prompt,
                "seed": seed if seed is not None else -1
            }
