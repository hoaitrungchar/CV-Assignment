import cv2
import numpy as np
from typing import Tuple, Optional

def process_video_frame( target_frame: np.ndarray,
    replacement_frame: np.ndarray,
    mask: np.ndarray,
    blend_width: int = 5
) -> np.ndarray:
    """
    Replace a masked region in the target video frame with the corresponding region
    from the replacement video frame.
    
    Args:
        target_frame: Frame from the target video where replacement will occur
        replacement_frame: Frame from the replacement video
        mask: Binary mask where white (255) indicates the region to replace
        blend_width: Width of the blending border in pixels
    
    Returns:
        The processed frame with the replacement region
    """
    # Ensure all inputs have the same dimensions
    if replacement_frame.shape[:2] != target_frame.shape[:2]:
        replacement_frame = cv2.resize(
            replacement_frame, 
            (target_frame.shape[1], target_frame.shape[0])
        )
    
    if mask.shape[:2] != target_frame.shape[:2]:
        mask = cv2.resize(
            mask, 
            (target_frame.shape[1], target_frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Ensure mask is binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Create blending border
    kernel = np.ones((blend_width, blend_width), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel)
    mask_eroded = cv2.erode(mask, kernel)
    blend_region = mask_dilated - mask_eroded
    
    # Create distance transform for smooth blending
    dist_transform = cv2.distanceTransform(mask_dilated, cv2.DIST_L2, 5)
    dist_transform = dist_transform * (blend_region > 0)
    dist_transform = dist_transform / (dist_transform.max() + 1e-6)
    
    # Stack masks for 3 channels
    blend_mask = np.stack([dist_transform] * 3, axis=-1)
    binary_mask = np.stack([mask > 0] * 3, axis=-1)
    
    # Combine frames with blending
    result = (replacement_frame * binary_mask + 
             replacement_frame * (1 - binary_mask) * blend_mask + 
             target_frame * (1 - binary_mask) * (1 - blend_mask)).astype(np.uint8)
    
    return result