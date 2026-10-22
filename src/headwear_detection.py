"""
Headwear Detection and Removal Mask Generation Module for HairTryOn.

Supports:
1. ResNet18 CelebAMask-HQ ONNX semantic segmentation (Class 18 = 'hat' / headwear)
2. Compatible cranial & headwear heuristic detector for caps/hats missed by the semantic parser
3. User-provided manual headwear / stage 1 removal mask override
4. High-contrast removal overlay generation for visual verification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# CelebAMask-HQ ONNX class 18 is 'hat' (headwear, caps, beanies)
HAT_SEG_CLASS = 18
HAIR_SEG_CLASS = 17
FACE_SKIN_SEG_CLASS = 1
EYEBROW_SEG_CLASSES = (2, 3)
BACKGROUND_SEG_CLASS = 0


@dataclass
class HeadwearDetectionResult:
    """Result of headwear detection containing masks, source, and metrics."""

    headwear_mask: Image.Image
    detected: bool
    source: str  # 'manual_mask', 'onnx_class_18', 'heuristic_detector', or 'none'
    coverage_pct: float
    pixel_count: int
    warnings: List[str]


def load_manual_mask(
    mask_source: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Load and format a manual mask into a binary uint8 array at target_size (w, h)."""
    w, h = target_size
    mask_np: Optional[np.ndarray] = None

    if isinstance(mask_source, (str, Path)):
        p = Path(mask_source)
        if not p.is_file():
            logger.warning(f"Manual headwear mask file not found: {p}")
            return None
        try:
            with Image.open(p) as pil_img:
                pil_gray = pil_img.convert("L")
                if pil_gray.size != (w, h):
                    pil_gray = pil_gray.resize((w, h), Image.Resampling.NEAREST)
                mask_np = np.asarray(pil_gray, dtype=np.uint8)
        except Exception as e:
            logger.error(f"Failed to load manual mask from {p}: {e}")
            return None
    elif isinstance(mask_source, Image.Image):
        pil_gray = mask_source.convert("L")
        if pil_gray.size != (w, h):
            pil_gray = pil_gray.resize((w, h), Image.Resampling.NEAREST)
        mask_np = np.asarray(pil_gray, dtype=np.uint8)
    elif isinstance(mask_source, np.ndarray):
        arr = mask_source.squeeze()
        if arr.shape[:2] != (h, w):
            arr = cv2.resize(arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask_np = arr.astype(np.uint8)

    if mask_np is not None:
        return (mask_np > 128).astype(np.uint8) * 255
    return None


def detect_headwear(
    image: Union[Image.Image, np.ndarray],
    seg_map: Optional[np.ndarray] = None,
    face_landmarks: Optional[Dict[str, Any]] = None,
    hair_mask: Optional[Union[Image.Image, np.ndarray]] = None,
    manual_mask: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
    enable_heuristic: bool = True,
) -> HeadwearDetectionResult:
    """
    Detect headwear (caps, hats, beanies) using segmentation, heuristic fallback, or manual mask.

    Args:
        image: Target portrait (PIL Image or BGR/RGB numpy array).
        seg_map: Optional 19-class ONNX segmentation map.
        face_landmarks: Optional MediaPipe 478-landmark dictionary.
        hair_mask: Optional hair mask to avoid double-counting.
        manual_mask: Optional explicit headwear mask file path or image.
        enable_heuristic: Whether to run color/spatial heuristic detector if ONNX misses headwear.

    Returns:
        HeadwearDetectionResult with binary headwear mask and detection metadata.
    """
    if isinstance(image, Image.Image):
        pil_rgb = image.convert("RGB")
        w, h = pil_rgb.size
        rgb_np = np.asarray(pil_rgb)
    else:
        h, w = image.shape[:2]
        rgb_np = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_rgb = Image.fromarray(rgb_np, mode="RGB")

    total_pixels = w * h
    warnings: List[str] = []

    # 1. Check for manual mask first
    if manual_mask is not None:
        loaded_manual = load_manual_mask(manual_mask, (w, h))
        if loaded_manual is not None and np.count_nonzero(loaded_manual) > 0:
            px = int(np.count_nonzero(loaded_manual))
            cov = (px / total_pixels) * 100.0
            return HeadwearDetectionResult(
                headwear_mask=Image.fromarray(loaded_manual, mode="L"),
                detected=True,
                source="manual_mask",
                coverage_pct=cov,
                pixel_count=px,
                warnings=warnings,
            )

    # 2. Check ONNX Segmentation Class 18 ('hat')
    if seg_map is not None:
        sm = seg_map
        if sm.shape[:2] != (h, w):
            sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST)
        onnx_hat_binary = (sm == HAT_SEG_CLASS).astype(np.uint8) * 255
        onnx_px = int(np.count_nonzero(onnx_hat_binary))
        if onnx_px > 100:
            cov = (onnx_px / total_pixels) * 100.0
            return HeadwearDetectionResult(
                headwear_mask=Image.fromarray(onnx_hat_binary, mode="L"),
                detected=True,
                source="onnx_class_18",
                coverage_pct=cov,
                pixel_count=onnx_px,
                warnings=warnings,
            )

    # 3. Compatible Heuristic Headwear Detector
    # When a cap/hat is worn, it sits on the upper cranial vault above the eyebrows and forehead.
    # If the parser classified part of it as background or skin, we analyze the upper cranial dome.
    if enable_heuristic and face_landmarks and face_landmarks.get("detected"):
        lm_pts = np.array(face_landmarks.get("landmarks_2d", []), dtype=np.int32)
        if len(lm_pts) >= 10:
            # MediaPipe eyebrow indices
            eyebrow_indices = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
            valid_indices = [idx for idx in eyebrow_indices if idx < len(lm_pts)]
            if valid_indices:
                eyebrow_pts = lm_pts[valid_indices]
                min_brow_y = int(np.min(eyebrow_pts[:, 1]))
                brow_x_min = int(np.min(eyebrow_pts[:, 0]))
                brow_x_max = int(np.max(eyebrow_pts[:, 0]))
                face_w = brow_x_max - brow_x_min

                # Check region above eyebrows:
                # Top boundary: 0 to min_brow_y
                # Horizontal range: expanded by 30% of face width on each side
                x_start = max(0, int(brow_x_min - 0.35 * face_w))
                x_end = min(w, int(brow_x_max + 0.35 * face_w))
                y_start = 0
                y_end = max(0, min_brow_y - 2)

                if y_end > y_start and x_end > x_start:
                    roi_rgb = rgb_np[y_start:y_end, x_start:x_end]

                    # Existing hair mask in ROI
                    existing_hair_roi = np.zeros((y_end - y_start, x_end - x_start), dtype=bool)
                    if hair_mask is not None:
                        h_np = np.asarray(hair_mask.convert("L") if isinstance(hair_mask, Image.Image) else hair_mask)
                        if h_np.shape[:2] != (h, w):
                            h_np = cv2.resize(h_np, (w, h), interpolation=cv2.INTER_NEAREST)
                        existing_hair_roi = h_np[y_start:y_end, x_start:x_end] > 128

                    # Check for non-hair, non-skin, non-background cap structure
                    # Backdrops in portraits are typically uniform, whereas hats have texture or distinct color
                    # Convert to HSV to detect hat color consistency
                    hsv_roi = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
                    # Detect if there is a dominant non-backdrop cluster above the forehead that isn't hair
                    # Only activate if there is significant non-hair matter above the forehead
                    # E.g. beige cap: Hue 15-40, Sat 25-180, Val 70-230
                    beige_cap_mask = cv2.inRange(hsv_roi, np.array([10, 20, 50]), np.array([45, 180, 240]))
                    cap_candidates = beige_cap_mask & (~existing_hair_roi)

                    # Exclude background wall and skin if segmentation map is available
                    if seg_map is not None:
                        sm = seg_map
                        if sm.shape[:2] != (h, w):
                            sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST)
                        seg_roi = sm[y_start:y_end, x_start:x_end]
                        is_bg_or_skin = (seg_roi == BACKGROUND_SEG_CLASS) | (seg_roi == FACE_SKIN_SEG_CLASS)
                        cap_candidates = cap_candidates & (~is_bg_or_skin)

                    # Remove noise with morphological opening
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    cap_candidates = cv2.morphologyEx(cap_candidates, cv2.MORPH_OPEN, kernel)

                    # Headwear MUST be contiguous with or adjacent to the head/hair region
                    # An isolated patch floating in mid-air far above the head is wall texture/watermark, not headwear
                    near_head = cv2.dilate(
                        existing_hair_roi.astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
                    ) > 0

                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cap_candidates)
                    headwear_roi = np.zeros_like(cap_candidates)

                    for i in range(1, num_labels):
                        area = stats[i, cv2.CC_STAT_AREA]
                        comp_mask = labels == i
                        if area > 800 and np.any(comp_mask & near_head):
                            headwear_roi[comp_mask] = 255

                    if np.count_nonzero(headwear_roi) > 800:
                        full_headwear = np.zeros((h, w), dtype=np.uint8)
                        full_headwear[y_start:y_end, x_start:x_end] = headwear_roi
                        px = int(np.count_nonzero(full_headwear))
                        cov = (px / total_pixels) * 100.0
                        warnings.append(f"Heuristic headwear detector found {px} potential cap/hat pixels.")
                        return HeadwearDetectionResult(
                            headwear_mask=Image.fromarray(full_headwear, mode="L"),
                            detected=True,
                            source="heuristic_detector",
                            coverage_pct=cov,
                            pixel_count=px,
                            warnings=warnings,
                        )

    # 4. No headwear detected
    empty_mask = np.zeros((h, w), dtype=np.uint8)
    return HeadwearDetectionResult(
        headwear_mask=Image.fromarray(empty_mask, mode="L"),
        detected=False,
        source="none",
        coverage_pct=0.0,
        pixel_count=0,
        warnings=warnings,
    )


def create_removal_overlay(
    image: Image.Image,
    removal_mask: Image.Image,
    color: Tuple[int, int, int] = (255, 30, 30),
    alpha: float = 0.50,
) -> Image.Image:
    """
    Generate a high-contrast visual overlay highlighting exactly what Stage 1 will remove.

    Args:
        image: Original portrait image.
        removal_mask: Binary mask of pixels to be removed (hair + headwear + fringe).
        color: RGB tuple for the translucent highlight (default: crimson red).
        alpha: Opacity of the highlight (0.0 to 1.0).

    Returns:
        PIL Image with highlighted removal zone and crisp boundary contour.
    """
    rgb = np.asarray(image.convert("RGB")).copy()
    h, w = rgb.shape[:2]

    mask_np = np.asarray(removal_mask.convert("L"))
    if mask_np.shape[:2] != (h, w):
        mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

    binary = (mask_np > 128).astype(np.uint8)

    # 1. Translucent fill
    tint = np.zeros_like(rgb)
    tint[:] = color
    mask_indices = binary > 0
    rgb[mask_indices] = (
        (1.0 - alpha) * rgb[mask_indices] + alpha * tint[mask_indices]
    ).astype(np.uint8)

    # 2. Crisp boundary contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (255, 255, 255), thickness=2)
    cv2.drawContours(rgb, contours, -1, color, thickness=1)

    return Image.fromarray(rgb, mode="RGB")
