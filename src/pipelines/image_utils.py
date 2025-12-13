"""
Image processing utilities for SAM Edit pipeline.

Provides blending functions for seamlessly compositing edited regions
back into original images.
"""

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from typing import Tuple


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format (BGR).

    Args:
        image: PIL Image in RGB mode

    Returns:
        numpy array in BGR format (OpenCV standard)
    """
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy array (RGB)
    rgb_array = np.array(image)

    # Convert RGB to BGR for OpenCV
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    return bgr_array


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV image (BGR) to PIL Image (RGB).

    Args:
        image: numpy array in BGR format

    Returns:
        PIL Image in RGB mode
    """
    # Convert BGR to RGB
    rgb_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_array)

    return pil_image


def create_feathered_mask(mask: np.ndarray, feather_radius: int = 10) -> np.ndarray:
    """
    Apply Gaussian blur to mask for smooth blending.

    Args:
        mask: Binary mask as numpy array (0-255 or 0-1)
        feather_radius: Blur radius in pixels

    Returns:
        Feathered mask normalized to 0-1 range
    """
    # Ensure mask is float in 0-1 range
    if mask.max() > 1.0:
        mask = mask.astype(np.float32) / 255.0
    else:
        mask = mask.astype(np.float32)

    # Apply Gaussian blur
    feathered = gaussian_filter(mask, sigma=feather_radius)

    # Normalize to 0-1
    if feathered.max() > 0:
        feathered = feathered / feathered.max()

    return feathered


def poisson_blend(
    source: Image.Image,
    target: Image.Image,
    mask: Image.Image,
    center: Tuple[int, int]
) -> Image.Image:
    """
    Poisson blending using cv2.seamlessClone for seamless compositing.

    This uses the Poisson equation to preserve gradients from the source
    while blending into the target, resulting in natural-looking composites.

    Args:
        source: Edited region (PIL Image)
        target: Original full image (PIL Image)
        mask: Binary mask for blending region (PIL Image, grayscale)
        center: (x, y) center point for cloning

    Returns:
        Blended PIL Image

    Raises:
        ValueError: If blending fails (size mismatch, invalid mask, etc.)
    """
    try:
        # Convert to OpenCV format
        src_cv2 = pil_to_cv2(source)
        tgt_cv2 = pil_to_cv2(target)

        # Ensure mask is grayscale and uint8
        if mask.mode != 'L':
            mask = mask.convert('L')
        mask_cv2 = np.array(mask, dtype=np.uint8)

        # Ensure images are same size
        if src_cv2.shape[:2] != tgt_cv2.shape[:2]:
            raise ValueError(
                f"Source and target must be same size. "
                f"Got source={src_cv2.shape[:2]}, target={tgt_cv2.shape[:2]}"
            )

        # Perform Poisson blending
        result = cv2.seamlessClone(
            src_cv2,
            tgt_cv2,
            mask_cv2,
            center,
            cv2.NORMAL_CLONE
        )

        # Convert back to PIL
        return cv2_to_pil(result)

    except Exception as e:
        raise ValueError(f"Poisson blending failed: {str(e)}")


def feather_blend(
    source: Image.Image,
    target: Image.Image,
    mask: Image.Image,
    feather_radius: int = 10
) -> Image.Image:
    """
    Feather blending with Gaussian blur for soft edge transitions.

    Simpler and faster than Poisson blending, uses alpha compositing
    with a feathered mask for smooth blending.

    Args:
        source: Edited region (PIL Image)
        target: Original full image (PIL Image)
        mask: Binary mask for blending region (PIL Image, grayscale)
        feather_radius: Blur radius for mask edges (pixels)

    Returns:
        Blended PIL Image
    """
    # Convert to numpy arrays
    src_array = np.array(source.convert('RGB'), dtype=np.float32)
    tgt_array = np.array(target.convert('RGB'), dtype=np.float32)

    # Ensure mask is grayscale
    if mask.mode != 'L':
        mask = mask.convert('L')
    mask_array = np.array(mask, dtype=np.float32) / 255.0

    # Create feathered mask
    feathered = create_feathered_mask(mask_array, feather_radius)

    # Expand mask to 3 channels (RGB)
    feathered_3ch = np.stack([feathered, feathered, feathered], axis=2)

    # Alpha composite: result = source * mask + target * (1 - mask)
    result = (
        src_array * feathered_3ch +
        tgt_array * (1.0 - feathered_3ch)
    )

    # Convert back to uint8 and PIL
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result, mode='RGB')


def simple_paste_blend(
    source: Image.Image,
    target: Image.Image,
    position: Tuple[int, int],
    mask: Image.Image = None
) -> Image.Image:
    """
    Simple paste operation as fallback when advanced blending fails.

    Args:
        source: Image to paste
        target: Target image to paste onto
        position: (x, y) top-left position
        mask: Optional mask for paste operation

    Returns:
        Target image with source pasted
    """
    result = target.copy()

    if mask is not None:
        # Ensure mask is grayscale
        if mask.mode != 'L':
            mask = mask.convert('L')
        result.paste(source, position, mask)
    else:
        result.paste(source, position)

    return result
