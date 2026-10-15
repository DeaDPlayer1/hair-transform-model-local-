"""
Geometry-Aware Hairstyle Alignment Module.

Aligns the reference hairstyle to the target head geometry using
MediaPipe face landmarks and a validated similarity transform.

Coordinate spaces (kept explicit throughout):
    - target_space:    original target image pixel coordinates (H_t × W_t)
    - reference_space: original reference image pixel coordinates (H_r × W_r)
    - aligned_space:   reference hair warped into target_space (H_t × W_t)

Outputs three separate mask objects:
    1. aligned_hair_alpha  – reference hair alpha warped into target_space
    2. target_edit_mask    – union of old target hair ∪ aligned reference hair,
                             minus protected facial features (binary 0/255)
    3. compositing_alpha   – inner-distance-transform feathered blend mask

Does NOT:
    - Treat landmark 10 as the exact hairline or skull top.
    - Use full affine (which would shear curls).
    - Indiscriminately subtract the entire forehead from the edit mask.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image

from src.face_landmarks import (
    detect_face_landmarks,
    get_mirrored_landmarks,
    build_symmetric_correspondence,
    analyze_pose_and_mirroring,
    create_head_constraints,
    BILATERAL_SYMMETRIC_PAIRS,
)
from src.hair_mask_refinement import (
    build_feathered_blend_mask,
    PROTECTED_SEG_CLASSES,
    CLOTHING_SEG_CLASSES,
)


# ─── Configuration ────────────────────────────────────────────────────

# Geometry anchors used for the similarity transform.
# These landmarks define HEAD SHAPE geometry (width, vertical extent),
# not the hairline.  Landmark 10 (forehead midline) is used as one
# geometric control point for vertical positioning, not as the hairline.
ALIGNMENT_ANCHOR_INDICES = [
    # Forehead / cranial geometry (NOT hairline)
    10,          # Forehead midline — vertical head top reference
    109, 338,    # Forehead lateral — head width above brows
    # Temples and ears — head width
    234, 454,    # Tragus / ear / temple
    103, 332,    # Temple boundary
    # Eyebrows — lower bound of hair region
    70, 300,     # Outer eyebrow
    105, 334,    # Mid eyebrow
    # Stable midline geometry
    168,         # Nasion (nose bridge top)
    8,           # Mid-forehead
]

# Validation thresholds for the similarity transform
MIN_INLIERS = 4              # Minimum RANSAC inliers for a valid transform
MIN_SCALE = 0.1              # Reject transforms that shrink hair below 10% (allows cross-resolution portraits)
MAX_SCALE = 4.0              # Reject transforms that enlarge hair above 400%
MAX_ROTATION_DEG = 45.0      # Reject rotations beyond ±45°
MIN_CONFIDENCE = 0.4         # Minimum inlier ratio for acceptable alignment

# Hairline transition zone — pixels above eyebrow midline where hair is
# allowed to fade in gradually rather than being hard-clipped.
HAIRLINE_TRANSITION_PX = 15


# ─── Transform Estimation ────────────────────────────────────────────

def _extract_anchor_points(
    landmarks_2d: np.ndarray,
    anchor_indices: list = ALIGNMENT_ANCHOR_INDICES,
) -> np.ndarray:
    """Extract anchor landmark coordinates as (N, 2) float32 array."""
    valid = [i for i in anchor_indices if i < len(landmarks_2d)]
    return landmarks_2d[valid].astype(np.float32)


def estimate_head_geometry(landmarks_2d: np.ndarray) -> Dict[str, Any]:
    """
    Estimate head geometry from face landmarks.

    Returns measurable head properties — NOT the hairline position.
    The hairline is estimated separately from the hair segmentation mask.
    """
    if landmarks_2d is None or len(landmarks_2d) < 200:
        return {"valid": False}

    # Head width from temples/ears
    left_temple = landmarks_2d[234]
    right_temple = landmarks_2d[454]
    head_width = float(np.linalg.norm(right_temple - left_temple))

    # Vertical span: forehead midline to chin
    forehead_top = landmarks_2d[10]  # geometric reference, not hairline
    chin = landmarks_2d[152]
    head_height = float(np.linalg.norm(chin - forehead_top))

    # Face center (nasion)
    nasion = landmarks_2d[168]

    # Eyebrow midline — lower bound of natural hair region
    brow_pts = landmarks_2d[[70, 105, 63, 300, 334, 296]]
    brow_y_min = float(brow_pts[:, 1].min())
    brow_y_mean = float(brow_pts[:, 1].mean())

    # Inter-eye distance for scale reference
    left_eye_outer = landmarks_2d[33]
    right_eye_outer = landmarks_2d[263]
    interocular_dist = float(np.linalg.norm(right_eye_outer - left_eye_outer))

    return {
        "valid": True,
        "head_width": head_width,
        "head_height": head_height,
        "nasion": nasion.astype(np.float32),
        "forehead_top_pt": forehead_top.astype(np.float32),
        "chin_pt": chin.astype(np.float32),
        "left_temple": left_temple.astype(np.float32),
        "right_temple": right_temple.astype(np.float32),
        "brow_y_min": brow_y_min,
        "brow_y_mean": brow_y_mean,
        "interocular_dist": interocular_dist,
    }


def compute_similarity_transform(
    src_landmarks_2d: np.ndarray,
    dst_landmarks_2d: np.ndarray,
    is_src_mirrored: bool = False,
    anchor_indices: list = ALIGNMENT_ANCHOR_INDICES,
) -> Dict[str, Any]:
    """
    Compute a validated similarity transform (scale + rotation + translation)
    mapping source (reference) landmarks to destination (target) landmarks.

    Uses cv2.estimateAffinePartial2D with RANSAC for robustness.

    Returns:
        dict with:
            - 'valid': bool
            - 'matrix': 2×3 np.ndarray (or None)
            - 'scale': float
            - 'rotation_deg': float
            - 'inliers': int
            - 'confidence': float (inlier ratio)
            - 'rejection_reason': str or None
    """
    result = {
        "valid": False,
        "matrix": None,
        "scale": 1.0,
        "rotation_deg": 0.0,
        "inliers": 0,
        "confidence": 0.0,
        "rejection_reason": None,
    }

    # Filter anchor indices to those within bounds for both arrays
    max_idx = min(len(src_landmarks_2d), len(dst_landmarks_2d))
    valid_anchors = [i for i in anchor_indices if i < max_idx]

    # Also check bilateral swapped indices for mirrored case
    if is_src_mirrored:
        valid_anchors = [
            i for i in valid_anchors
            if BILATERAL_SYMMETRIC_PAIRS.get(i, i) < len(dst_landmarks_2d)
        ]

    if len(valid_anchors) < 3:
        result["rejection_reason"] = (
            f"Insufficient valid anchor indices: {len(valid_anchors)} < 3 "
            f"(landmark arrays have {max_idx} points)"
        )
        return result

    # Build corresponding point pairs
    src_pts, dst_pts = build_symmetric_correspondence(
        src_landmarks_2d, dst_landmarks_2d,
        is_src_mirrored=is_src_mirrored,
        anchor_indices=valid_anchors,
    )

    if len(src_pts) < 3:
        result["rejection_reason"] = (
            f"Insufficient correspondences: {len(src_pts)} < 3 minimum"
        )
        return result

    # Estimate similarity transform with RANSAC
    M, inlier_mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=8.0,
        maxIters=2000,
        confidence=0.99,
    )

    if M is None:
        result["rejection_reason"] = "RANSAC failed to find a valid transform"
        return result

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    confidence = inliers / max(1, len(src_pts))

    # Decompose similarity matrix [[a, -b, tx], [b, a, ty]]
    a, b = M[0, 0], M[1, 0]
    scale = float(np.sqrt(a * a + b * b))
    rotation_deg = float(np.degrees(np.arctan2(b, a)))
    tx = float(M[0, 2])
    ty = float(M[1, 2])

    result["mode"] = "similarity"
    result["matrix"] = M
    result["scale"] = scale
    result["rotation_deg"] = rotation_deg
    result["tx"] = tx
    result["ty"] = ty
    result["inliers"] = inliers
    result["confidence"] = confidence

    # ── Validation gates ──

    if inliers < MIN_INLIERS:
        result["rejection_reason"] = (
            f"Too few RANSAC inliers: {inliers} < {MIN_INLIERS}"
        )
        return result

    if confidence < MIN_CONFIDENCE:
        result["rejection_reason"] = (
            f"Low alignment confidence: {confidence:.2f} < {MIN_CONFIDENCE}"
        )
        return result

    if scale < MIN_SCALE or scale > MAX_SCALE:
        result["rejection_reason"] = (
            f"Implausible scale: {scale:.2f} "
            f"(valid range: {MIN_SCALE}–{MAX_SCALE})"
        )
        return result

    if abs(rotation_deg) > MAX_ROTATION_DEG:
        result["rejection_reason"] = (
            f"Excessive rotation: {rotation_deg:.1f}° "
            f"(max ±{MAX_ROTATION_DEG}°)"
        )
        return result

    result["valid"] = True
    return result


# ─── Fallback Transform ──────────────────────────────────────────────

def compute_fallback_transform(
    target_geometry: Dict[str, Any],
    reference_geometry: Dict[str, Any],
    target_hair_mask_np: np.ndarray,
    ref_hair_mask_np: np.ndarray,
    target_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """
    Compute a simple center-and-scale fallback when landmark-based
    similarity transform is rejected or unavailable.

    Strategy: match the hair bounding-box center and scale based on
    head width ratio (or mask bounding-box width if geometry is missing).
    """
    h_t, w_t = target_shape[:2]

    # Target hair centroid and bounding box
    t_ys, t_xs = np.where(target_hair_mask_np > 128)
    if len(t_xs) == 0:
        # No target hair — center on face or image center
        t_cx = w_t / 2.0
        t_cy = h_t / 4.0  # upper quarter
        t_bbox_w = w_t * 0.5
    else:
        t_cx = float(t_xs.mean())
        t_cy = float(t_ys.mean())
        t_bbox_w = float(t_xs.max() - t_xs.min() + 1)

    # Reference hair centroid and bounding box
    r_ys, r_xs = np.where(ref_hair_mask_np > 128)
    if len(r_xs) == 0:
        return {
            "valid": False,
            "matrix": None,
            "scale": 1.0,
            "rotation_deg": 0.0,
            "inliers": 0,
            "confidence": 0.0,
            "rejection_reason": "Reference hair mask is empty",
            "is_fallback": True,
        }

    r_cx = float(r_xs.mean())
    r_cy = float(r_ys.mean())
    r_bbox_w = float(r_xs.max() - r_xs.min() + 1)

    # Scale: prefer head-width ratio if geometry is available
    if (target_geometry.get("valid") and reference_geometry.get("valid")
            and target_geometry["head_width"] > 30
            and reference_geometry["head_width"] > 30):
        scale = target_geometry["head_width"] / reference_geometry["head_width"]
    else:
        scale = max(0.5, min(2.5, t_bbox_w / max(1, r_bbox_w)))

    # Clamp scale
    scale = max(MIN_SCALE, min(MAX_SCALE, scale))

    # Build similarity matrix: translate ref centroid to target centroid with scale
    tx = t_cx - scale * r_cx
    ty = t_cy - scale * r_cy
    M = np.array([
        [scale, 0.0, tx],
        [0.0, scale, ty],
    ], dtype=np.float64)

    return {
        "valid": True,
        "mode": "fallback_centroid",
        "matrix": M,
        "scale": scale,
        "rotation_deg": 0.0,
        "tx": float(tx),
        "ty": float(ty),
        "inliers": 0,
        "confidence": 0.0,
        "rejection_reason": None,
        "is_fallback": True,
    }


# ─── Hair Warping ─────────────────────────────────────────────────────

def warp_reference_hair(
    reference_rgb: np.ndarray,
    reference_alpha: np.ndarray,
    transform_matrix: np.ndarray,
    target_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp reference hair RGB and alpha into target image coordinates
    using the SAME geometric transform.

    Args:
        reference_rgb:   (H_r, W_r, 3) uint8 — reference hair pixels on neutral bg
        reference_alpha: (H_r, W_r) float32 [0, 1] — reference hair soft alpha
        transform_matrix: 2×3 similarity/affine matrix
        target_shape:    (H_t, W_t) — output dimensions

    Returns:
        (warped_rgb, warped_alpha):
            warped_rgb:   (H_t, W_t, 3) uint8
            warped_alpha: (H_t, W_t) float32 [0, 1]

    Interpolation:
        - RGB uses INTER_LINEAR for smooth color blending
        - Alpha uses INTER_LINEAR to preserve soft curl edges
          (INTER_NEAREST would jaggify fine strands)
    """
    h_t, w_t = target_shape[:2]

    # Premultiplied alpha warping prevents edge-fringe color contamination
    premul_rgb = reference_rgb.astype(np.float32) * reference_alpha[..., None]
    warped_premul = cv2.warpAffine(
        premul_rgb,
        transform_matrix,
        (w_t, h_t),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    warped_alpha = cv2.warpAffine(
        reference_alpha,
        transform_matrix,
        (w_t, h_t),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    # Clamp alpha to valid range after interpolation
    warped_alpha = np.clip(warped_alpha, 0.0, 1.0).astype(np.float32)

    # De-premultiply to recover clean foreground RGB where alpha is positive,
    # falling back to raw warped RGB where alpha is zero (for consistency on zero-opacity pixels)
    warped_rgb_raw = cv2.warpAffine(
        reference_rgb,
        transform_matrix,
        (w_t, h_t),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    warped_rgb = warped_rgb_raw.copy()
    valid = warped_alpha > 1e-5
    warped_rgb[valid] = np.clip(
        np.round(warped_premul[valid] / warped_alpha[valid, None]), 0, 255
    ).astype(np.uint8)

    return warped_rgb, warped_alpha


# ─── Protected Region Mask ────────────────────────────────────────────

def build_facial_protection_mask(
    target_shape: Tuple[int, int],
    target_landmarks: Optional[Dict[str, Any]] = None,
    target_segmentation_map: Optional[np.ndarray] = None,
    hairline_transition_px: int = HAIRLINE_TRANSITION_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a facial protection mask and a hairline transition zone.

    Returns:
        (hard_protection, transition_weight):
            hard_protection:  (H, W) uint8, 255 = strictly protected
            transition_weight: (H, W) float32, 0–1 ramp near the hairline
                               1.0 = fully allow hair, 0.0 = protected
    """
    h, w = target_shape[:2]
    hard_protection = np.zeros((h, w), dtype=np.uint8)
    transition_weight = np.ones((h, w), dtype=np.float32)

    # Resolution-adaptive kernel
    k_size = max(3, int(round(h * 0.006)))
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    # Layer 1: Segmentation-based protection
    if target_segmentation_map is not None:
        seg = target_segmentation_map
        if seg.shape[:2] != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)

        # Hard-protect eyes, eyebrows, nose, mouth, ears
        feat_mask = np.isin(seg, PROTECTED_SEG_CLASSES).astype(np.uint8) * 255
        feat_mask = cv2.dilate(feat_mask, kernel, iterations=1)
        hard_protection = np.maximum(hard_protection, feat_mask)

        # Hard-protect clothing
        cloth_mask = np.isin(seg, CLOTHING_SEG_CLASSES).astype(np.uint8) * 255
        hard_protection = np.maximum(hard_protection, cloth_mask)

    # Layer 2: Landmark-based protection (redundant safeguard)
    if target_landmarks and target_landmarks.get("detected"):
        lms = target_landmarks["landmarks_2d"]
        pts = np.int32(lms)

        # Eye hulls
        left_eye_idx = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145]
        right_eye_idx = [263, 362, 387, 386, 385, 384, 398, 380, 373, 374]
        # Mouth hull
        mouth_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                     308, 185, 40, 39, 37, 0, 267, 269, 270, 409]
        # Nose hull
        nose_idx = [168, 6, 197, 195, 5, 4, 1, 2, 98, 327]

        for idx_group in [left_eye_idx, right_eye_idx, mouth_idx, nose_idx]:
            valid_pts = [pts[i] for i in idx_group if i < len(pts)]
            if len(valid_pts) >= 3:
                hull = cv2.convexHull(np.array(valid_pts))
                cv2.fillConvexPoly(hard_protection, hull, 255)

        # Dilate landmark protections
        hard_protection = cv2.dilate(hard_protection, kernel, iterations=1)

    # ── Facial protection transition zone (smooth 2D distance transform) ──
    # Avoid 1D horizontal row-slicing which creates visible horizontal artifacts across the image.
    # Instead, compute a smooth, isotropic 2D distance transform from the protected facial boundaries.
    # At facial features (eyes/brows/mouth/nose), weight is 0.0.
    # Moving outward into forehead or hair, weight ramps smoothly to 1.0.
    # Regions away from the face (cranium, side hair) remain 1.0.
    if np.any(hard_protection > 0):
        non_protected = (hard_protection == 0).astype(np.uint8)
        dist = cv2.distanceTransform(non_protected, cv2.DIST_L2, 5)
        trans_px = max(4.0, float(hairline_transition_px))
        transition_weight = np.clip(dist / trans_px, 0.0, 1.0).astype(np.float32)

    return hard_protection, transition_weight


# ─── Three-Mask Construction ─────────────────────────────────────────

def build_alignment_masks(
    aligned_hair_alpha: np.ndarray,
    target_hair_mask_np: np.ndarray,
    hard_protection: np.ndarray,
    transition_weight: np.ndarray,
    target_shape: Tuple[int, int],
    cranial_expansion_px: int = 20,
    feather_width_px: float = 6.0,
) -> Dict[str, Any]:
    """
    Construct the three separate mask objects.

    1. aligned_hair_alpha — already computed, passed through with clipping
    2. target_edit_mask   — union of old hair ∪ aligned hair, minus protection
    3. compositing_alpha  — feathered blend mask from edit mask

    Args:
        aligned_hair_alpha: (H, W) float32 [0, 1] — warped reference hair
        target_hair_mask_np: (H, W) uint8 — original target hair mask (0/255)
        hard_protection: (H, W) uint8 — strictly protected regions (255)
        transition_weight: (H, W) float32 [0, 1] — hairline transition ramp
        target_shape: (H, W) of the target image
        cranial_expansion_px: pixels to dilate the union for volume headroom
        feather_width_px: inner feathering for compositing alpha

    Returns:
        dict with 'aligned_hair_alpha', 'target_edit_mask', 'compositing_alpha'
    """
    h, w = target_shape[:2]

    # Ensure dimensions match
    if aligned_hair_alpha.shape[:2] != (h, w):
        aligned_hair_alpha = cv2.resize(
            aligned_hair_alpha, (w, h), interpolation=cv2.INTER_LINEAR
        )
    if target_hair_mask_np.shape[:2] != (h, w):
        target_hair_mask_np = cv2.resize(
            target_hair_mask_np, (w, h), interpolation=cv2.INTER_NEAREST
        )

    # Apply hairline transition to aligned alpha
    aligned_clipped = aligned_hair_alpha * transition_weight
    aligned_clipped = np.clip(aligned_clipped, 0.0, 1.0).astype(np.float32)

    # Binary aligned region (threshold at 0.3 to capture soft edges)
    aligned_binary = (aligned_clipped > 0.3).astype(np.uint8) * 255

    # Union: old target hair ∪ aligned reference hair
    target_hair_binary = (target_hair_mask_np > 128).astype(np.uint8) * 255
    union_mask = np.maximum(target_hair_binary, aligned_binary)

    # Cranial expansion for new hairstyle volume
    if cranial_expansion_px > 0:
        k = max(3, cranial_expansion_px)
        if k % 2 == 0:
            k += 1
        expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        union_mask = cv2.dilate(union_mask, expand_kernel, iterations=1)

    # Subtract hard-protected regions
    union_mask[hard_protection > 0] = 0

    # Ensure binary
    edit_mask_np = (union_mask > 128).astype(np.uint8) * 255
    edit_mask_pil = Image.fromarray(edit_mask_np, mode="L")

    # Compositing alpha via feathered blend mask
    compositing_alpha = build_feathered_blend_mask(
        edit_mask_pil, feather_width_px=feather_width_px
    )

    return {
        "aligned_hair_alpha": aligned_clipped,
        "target_edit_mask": edit_mask_pil,
        "compositing_alpha": compositing_alpha,
    }


# ─── Hair Harmonization & Edge Repair Masks ──────────────────────────

def harmonize_hair_color(
    aligned_hair_rgb: np.ndarray,
    aligned_hair_alpha: np.ndarray,
    target_rgb: np.ndarray,
    target_hair_mask_np: np.ndarray,
    strength: float = 0.40,
) -> np.ndarray:
    """
    Match color and lighting temperature of aligned reference hair to target in CIE LAB space.
    Restrained matching preserves donor hair curl texture, highlights, and fine strands while
    harmonizing overall lightness and tone with the target portrait lighting.
    """
    # Restrict harmonization statistics and color transfer strictly to core interior hair
    # (alpha > 0.65) to prevent amplifying background fringe/halos at delicate boundary strands.
    core_indices = aligned_hair_alpha > 0.65
    if not np.any(core_indices):
        return aligned_hair_rgb.copy()

    # Convert RGB to LAB
    ref_bgr = cv2.cvtColor(aligned_hair_rgb, cv2.COLOR_RGB2BGR)
    tgt_bgr = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)

    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    tgt_hair_indices = target_hair_mask_np > 128
    if np.count_nonzero(tgt_hair_indices) > 50:
        tgt_sample = tgt_lab[tgt_hair_indices]
    else:
        # Fallback to upper 40% of portrait if target has minimal hair
        h = target_rgb.shape[0]
        tgt_sample = tgt_lab[: int(h * 0.4), :]

    ref_sample = ref_lab[core_indices]

    result_lab = ref_lab.copy()
    for ch in range(3):
        m_tgt = float(tgt_sample[:, ch].mean())
        s_tgt = float(max(1e-3, tgt_sample[:, ch].std()))

        m_ref = float(ref_sample[:, ch].mean())
        s_ref = float(max(1e-3, ref_sample[:, ch].std()))

        # Restrained ratio prevents blowing out contrast
        ratio = np.clip(s_tgt / s_ref, 0.75, 1.25)
        matched = (ref_lab[:, :, ch] - m_ref) * ratio + m_tgt
        result_lab[:, :, ch] = np.where(
            core_indices,
            (1.0 - strength) * ref_lab[:, :, ch] + strength * matched,
            ref_lab[:, :, ch],
        )

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def composite_aligned_hair(
    target_rgb: np.ndarray,
    aligned_hair_rgb: np.ndarray,
    aligned_hair_alpha: np.ndarray,
    hard_protection: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Composite aligned reference hair onto target image with facial feature protection.

    Returns:
        (pasted_composite_rgb, effective_alpha):
            pasted_composite_rgb: (H, W, 3) uint8 composite image
            effective_alpha: (H, W) float32 in [0, 1]
    """
    h, w = target_rgb.shape[:2]
    hair_rgb = aligned_hair_rgb
    hair_alpha = aligned_hair_alpha

    if hair_rgb.shape[:2] != (h, w):
        hair_rgb = cv2.resize(hair_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    if hair_alpha.shape[:2] != (h, w):
        hair_alpha = cv2.resize(hair_alpha, (w, h), interpolation=cv2.INTER_LINEAR)

    alpha = np.clip(hair_alpha, 0.0, 1.0).astype(np.float32)
    if hard_protection is not None:
        if hard_protection.shape[:2] != (h, w):
            hard_protection = cv2.resize(hard_protection, (w, h), interpolation=cv2.INTER_NEAREST)
        alpha[hard_protection > 0] = 0.0

    alpha_3ch = alpha[..., None]
    composite = (
        target_rgb.astype(np.float32) * (1.0 - alpha_3ch) +
        hair_rgb.astype(np.float32) * alpha_3ch
    ).clip(0, 255).astype(np.uint8)

    return composite, alpha


def build_edge_repair_mask(
    aligned_hair_alpha: np.ndarray,
    target_hair_mask_np: np.ndarray,
    hard_protection: Optional[np.ndarray],
    target_shape: Tuple[int, int],
    edge_width_px: int = 24,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Build a dedicated, narrow edge-repair mask around the hairstyle boundary.

    Guarantees:
    1. The central hair texture and curl pattern from the reference composite are EXCLUDED
       from inpainting (their mask value is 0), so diffusion does NOT repaint them.
    2. The boundary ribbon (hairline, temples, outer silhouette) is INCLUDED (255) for inpainting.
    3. Any uncovered old target hair sticking out around the new hairstyle is INCLUDED (255)
       so diffusion can remove it without leaving old curls underneath.
    4. Protected facial features (eyes, eyebrows, nose, mouth) and clothing are STRICTLY EXCLUDED (0).

    Returns:
        (edge_repair_mask_pil, edge_repair_mask_np)
    """
    h, w = target_shape[:2]
    alpha = aligned_hair_alpha
    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    tgt_mask = target_hair_mask_np
    if tgt_mask.shape[:2] != (h, w):
        tgt_mask = cv2.resize(tgt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Resolution-adaptive width
    scale_factor = max(0.5, min(h, w) / 512.0)
    w_px = max(5, int(round(edge_width_px * scale_factor)))
    if w_px % 2 == 0:
        w_px += 1
    k_band = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w_px, w_px))

    # A. Hair silhouette boundaries
    hair_core = (alpha > 0.70).astype(np.uint8) * 255
    hair_footprint = (alpha > 0.15).astype(np.uint8) * 255

    # Boundary band via morphological gradient
    dilated_footprint = cv2.dilate(hair_footprint, k_band, iterations=1)
    eroded_core = cv2.erode(hair_core, k_band, iterations=1)
    boundary_band = cv2.subtract(dilated_footprint, eroded_core)

    # B. Uncovered old target hair (where target had hair, but new hairstyle doesn't cover it)
    old_hair = (tgt_mask > 128).astype(np.uint8) * 255
    covered_by_new = (alpha > 0.30).astype(np.uint8) * 255
    uncovered_old_hair = cv2.subtract(old_hair, covered_by_new)

    # C. Combine boundary seam + uncovered old hair
    combined_mask = np.maximum(boundary_band, uncovered_old_hair)

    # D. Strictly protect central hair core (keep reference curls intact)
    central_protected_hair = cv2.subtract(eroded_core, uncovered_old_hair)
    combined_mask[central_protected_hair > 0] = 0

    # E. Strictly protect facial features and clothing
    if hard_protection is not None:
        prot = hard_protection
        if prot.shape[:2] != (h, w):
            prot = cv2.resize(prot, (w, h), interpolation=cv2.INTER_NEAREST)
        combined_mask[prot > 0] = 0

    edge_mask_np = (combined_mask > 128).astype(np.uint8) * 255
    edge_mask_pil = Image.fromarray(edge_mask_np, mode="L")
    return edge_mask_pil, edge_mask_np


# ─── Diagnostic Previews ─────────────────────────────────────────────

def create_alignment_previews(
    target_rgb: np.ndarray,
    aligned_hair_rgb: np.ndarray,
    aligned_hair_alpha: np.ndarray,
    target_edit_mask_np: np.ndarray,
    target_landmarks: Optional[Dict[str, Any]] = None,
    hard_protection: Optional[np.ndarray] = None,
    pasted_composite: Optional[np.ndarray] = None,
    edge_repair_mask_np: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate diagnostic preview images for inspection before diffusion.

    Returns dict of images ready for display or saving:
        - 'aligned_rgb_preview':  aligned hair RGB on neutral background
        - 'alpha_preview':        aligned hair alpha as grayscale (no seams)
        - 'pasted_composite':     aligned hair composited directly onto target portrait
        - 'edge_repair_mask':     narrow boundary inpainting mask (central hair protected)
        - 'overlay_preview':      aligned hair overlaid on target with landmarks
        - 'edit_mask_preview':    edge repair mask with protection boundaries
    """
    h, w = target_rgb.shape[:2]
    line_t = max(1, int(round(h * 0.004)))
    pt_r = max(2, int(round(h * 0.006)))

    previews = {}

    # 1. Aligned RGB preview (hair on neutral gray background)
    previews["aligned_rgb_preview"] = aligned_hair_rgb.copy()

    # 2. Alpha preview (grayscale)
    alpha_vis = (aligned_hair_alpha * 255).astype(np.uint8)
    previews["alpha_preview"] = alpha_vis

    # 3. Aligned hair composited directly onto target portrait
    if pasted_composite is not None:
        previews["pasted_composite"] = pasted_composite.copy()

    # 4. Narrow edge-repair mask
    if edge_repair_mask_np is not None:
        previews["edge_repair_mask"] = edge_repair_mask_np.copy()

    # 5. Overlay preview: semi-transparent hair on target + landmarks
    overlay = target_rgb.copy()
    alpha_3ch = aligned_hair_alpha[..., None]

    # Blend aligned hair onto target (cyan tint for visibility)
    hair_tint = aligned_hair_rgb.copy()
    blend_region = aligned_hair_alpha > 0.1
    if np.any(blend_region):
        overlay_color = np.array([180, 255, 220], dtype=np.uint8)  # light cyan-green
        tinted = (
            hair_tint.astype(np.float32) * 0.5 +
            overlay_color.astype(np.float32) * 0.5
        ).astype(np.uint8)

        alpha_exp = alpha_3ch * 0.55  # semi-transparent
        overlay = (
            overlay.astype(np.float32) * (1.0 - alpha_exp) +
            tinted.astype(np.float32) * alpha_exp
        ).clip(0, 255).astype(np.uint8)

    # Draw target landmarks
    if target_landmarks and target_landmarks.get("detected"):
        lms = target_landmarks["landmarks_2d"]

        # Hairline curve (yellow)
        hairline_idx = [234, 127, 162, 21, 54, 103, 67, 109,
                        10, 338, 297, 332, 284, 251, 389, 356, 454]
        valid_hl = [i for i in hairline_idx if i < len(lms)]
        if len(valid_hl) >= 3:
            pts_hl = lms[valid_hl].astype(np.int32)
            cv2.polylines(overlay, [pts_hl], False, (255, 255, 0), line_t + 1)

        # Eyebrows (blue in RGB: (0, 128, 255))
        l_idx = [i for i in [70, 63, 105, 66, 107] if i < len(lms)]
        r_idx = [i for i in [336, 296, 334, 293, 300] if i < len(lms)]
        if len(l_idx) >= 2:
            l_brow = lms[l_idx].astype(np.int32)
            cv2.polylines(overlay, [l_brow], False, (0, 128, 255), line_t)
        if len(r_idx) >= 2:
            r_brow = lms[r_idx].astype(np.int32)
            cv2.polylines(overlay, [r_brow], False, (0, 128, 255), line_t)

        # Eyes (cyan in RGB: (0, 255, 255))
        le_idx = [i for i in [33, 160, 158, 133, 153, 144, 33] if i < len(lms)]
        re_idx = [i for i in [362, 385, 387, 263, 373, 380, 362] if i < len(lms)]
        if len(le_idx) >= 3:
            cv2.polylines(overlay, [lms[le_idx].astype(np.int32)], True, (0, 255, 255), line_t)
        if len(re_idx) >= 3:
            cv2.polylines(overlay, [lms[re_idx].astype(np.int32)], True, (0, 255, 255), line_t)

        # Temples (magenta circles)
        for idx in [234, 454]:
            if idx < len(lms):
                pt = tuple(lms[idx].astype(np.int32))
                cv2.circle(overlay, pt, pt_r + 2, (255, 0, 255), -1)

    previews["overlay_preview"] = overlay

    # 6. Edit mask preview with protection boundary
    mask_to_show = edge_repair_mask_np if edge_repair_mask_np is not None else target_edit_mask_np
    mask_vis = cv2.cvtColor(mask_to_show, cv2.COLOR_GRAY2RGB)
    if hard_protection is not None:
        # Draw protection contour in red
        contours, _ = cv2.findContours(
            hard_protection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(mask_vis, contours, -1, (255, 0, 0), line_t)

    previews["edit_mask_preview"] = mask_vis

    return previews


def save_alignment_previews(
    previews: Dict[str, np.ndarray],
    output_dir: str,
) -> Dict[str, str]:
    """Save all preview images to disk and return their paths."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    for name, img in previews.items():
        path = out / f"{name}.png"
        if isinstance(img, Image.Image):
            img.save(path)
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                Image.fromarray(img).save(path)
            elif img.ndim == 3:
                Image.fromarray(img).save(path)
        saved_paths[name] = str(path)

    return saved_paths


# ─── Main Alignment Entry Point ──────────────────────────────────────

def align_reference_to_target(
    target_image: Image.Image,
    reference_image: Image.Image,
    target_hair_mask: Image.Image,
    reference_hair_mask: Image.Image,
    target_landmarks: Optional[Dict[str, Any]] = None,
    reference_landmarks: Optional[Dict[str, Any]] = None,
    target_segmentation_map: Optional[np.ndarray] = None,
    cranial_expansion_px: int = 20,
    feather_width_px: float = 6.0,
    hairline_transition_px: int = HAIRLINE_TRANSITION_PX,
    save_previews_dir: Optional[str] = None,
    bounded_decontamination: bool = True,
    ring_radius_px: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full alignment pipeline: detect landmarks, compute transform, warp hair,
    build masks, generate previews.

    Args:
        target_image: PIL RGB — the person whose face we preserve
        reference_image: PIL RGB — the hairstyle source
        target_hair_mask: PIL L — binary mask of target's current hair
        reference_hair_mask: PIL L — binary mask of reference hair
        target_landmarks: pre-computed target landmarks (or None to detect)
        reference_landmarks: pre-computed reference landmarks (or None to detect)
        target_segmentation_map: ONNX face-parsing map for protection
        cranial_expansion_px: edit mask expansion for volume
        feather_width_px: inner feathering width
        hairline_transition_px: soft transition zone above eyebrows
        save_previews_dir: if set, save diagnostic images here

    Returns:
        dict with:
            - 'aligned_hair_rgb': (H_t, W_t, 3) uint8
            - 'aligned_hair_alpha': (H_t, W_t) float32 [0, 1]
            - 'target_edit_mask': PIL Image mode 'L'
            - 'compositing_alpha': (H_t, W_t, 1) float32 [0, 1]
            - 'transform_info': similarity transform details
            - 'pose_info': mirroring and pose analysis
            - 'previews': dict of preview images (if generated)
            - 'preview_paths': dict of saved preview paths (if saved)
            - 'warnings': list of warning strings
    """
    warnings = []

    # ── Convert to numpy ──
    target_rgb = np.asarray(target_image.convert("RGB"), dtype=np.uint8)
    reference_rgb = np.asarray(reference_image.convert("RGB"), dtype=np.uint8)
    h_t, w_t = target_rgb.shape[:2]
    h_r, w_r = reference_rgb.shape[:2]
    target_shape = (h_t, w_t)

    target_hair_np = np.asarray(target_hair_mask.convert("L"), dtype=np.uint8)
    if target_hair_np.shape[:2] != (h_t, w_t):
        target_hair_np = cv2.resize(
            target_hair_np, (w_t, h_t), interpolation=cv2.INTER_NEAREST
        )

    ref_hair_np = np.asarray(reference_hair_mask.convert("L"), dtype=np.uint8)
    if ref_hair_np.shape[:2] != (h_r, w_r):
        ref_hair_np = cv2.resize(
            ref_hair_np, (w_r, h_r), interpolation=cv2.INTER_NEAREST
        )

    # ── Detect landmarks if not provided ──
    if target_landmarks is None or not target_landmarks.get("detected"):
        target_bgr = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)
        target_landmarks = detect_face_landmarks(target_bgr)
        if not target_landmarks.get("detected"):
            warnings.append("No face detected in target image.")

    if reference_landmarks is None or not reference_landmarks.get("detected"):
        reference_bgr = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR)
        reference_landmarks = detect_face_landmarks(reference_bgr)
        if not reference_landmarks.get("detected"):
            warnings.append("No face detected in reference image.")

    # ── Pose analysis and mirroring ──
    pose_info = analyze_pose_and_mirroring(target_landmarks, reference_landmarks)
    should_mirror = pose_info.get("recommend_mirror", False)
    if pose_info.get("pose_mismatch_warning"):
        warnings.append(pose_info["pose_mismatch_warning"])

    # Apply mirroring to reference if needed
    if should_mirror:
        reference_rgb = cv2.flip(reference_rgb, 1)
        ref_hair_np = cv2.flip(ref_hair_np, 1)
        if reference_landmarks.get("detected"):
            reference_landmarks = dict(reference_landmarks)
            reference_landmarks["landmarks_2d"] = get_mirrored_landmarks(
                reference_landmarks["landmarks_2d"], w_r
            )

    # ── Prune false-positive studio backdrop from reference hair mask ──
    # The ONNX face parser frequently labels bright studio backdrops in the gaps
    # and perimeter of curls as hair. Detect the backdrop color and prune false positives.
    border_px = max(10, int(round(15 * (max(h_r, w_r) / 512.0))))
    top_b = reference_rgb[:border_px, :]
    left_b = reference_rgb[:, :border_px]
    right_b = reference_rgb[:, -border_px:]
    perim_rgb = np.concatenate([top_b.reshape(-1, 3), left_b.reshape(-1, 3), right_b.reshape(-1, 3)], axis=0)
    perim_hair = np.concatenate([
        (ref_hair_np[:border_px, :] > 128).reshape(-1),
        (ref_hair_np[:, :border_px] > 128).reshape(-1),
        (ref_hair_np[:, -border_px:] > 128).reshape(-1)
    ])
    bg_perim = perim_rgb[~perim_hair]
    if len(bg_perim) > 30:
        backdrop_color = np.median(bg_perim, axis=0)
        backdrop_std = np.std(bg_perim, axis=0).max()
    else:
        corners = np.concatenate([
            reference_rgb[:25, :25],
            reference_rgb[:25, -25:]
        ], axis=1)
        backdrop_color = corners.mean(axis=(0, 1))
        backdrop_std = corners.std(axis=(0, 1)).max()

    color_dist = np.linalg.norm(reference_rgb.astype(np.float32) - backdrop_color, axis=-1)
    # Conservative threshold: only prune pixels within 3.5 std of the backdrop
    prune_threshold = max(24.0, float(backdrop_std * 3.5))
    is_backdrop = color_dist < prune_threshold

    # Protect deep interior hair core and genuine highlights from false pruning:
    core_deep_k = max(5, int(round(7 * (max(h_r, w_r) / 512.0))))
    if core_deep_k % 2 == 0:
        core_deep_k += 1
    hair_core_deep = cv2.erode((ref_hair_np > 128).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_deep_k, core_deep_k)))
    prune_mask = (ref_hair_np > 128) & is_backdrop & (hair_core_deep == 0)

    pruned_count = int(np.count_nonzero(prune_mask))
    if pruned_count > 0:
        warnings.append(f"Pruned {pruned_count} false-positive backdrop pixels from reference hair mask.")

    ref_hair_np_clean = ref_hair_np.copy()
    ref_hair_np_clean[prune_mask] = 0

    # ── Build reference hair alpha (soft-edge) ──
    # Use distance transform on clean mask to create soft edges for curls
    ref_hair_binary = (ref_hair_np_clean > 128).astype(np.uint8)
    ref_hair_dist = cv2.distanceTransform(ref_hair_binary, cv2.DIST_L2, 5)
    # Soft alpha: full opacity after 3px from edge, linear ramp at boundary
    ref_hair_alpha = np.clip(ref_hair_dist / 3.0, 0.0, 1.0).astype(np.float32)

    # ── Foreground color decontamination ──
    # Inpaint interior hair color into the boundary zone so semitransparent
    # alpha interpolation at curl tips blends pure hair color rather than studio backdrop.
    scale_ref = max(h_r, w_r) / 512.0
    erode_k = max(3, int(round(5.0 * scale_ref)))
    if erode_k % 2 == 0:
        erode_k += 1
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    core_hair = cv2.erode(ref_hair_binary, k_erode, iterations=1)
    if not np.any(core_hair > 0):
        k_erode_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        core_hair = cv2.erode(ref_hair_binary, k_erode_3, iterations=1)
        if not np.any(core_hair > 0):
            core_hair = ref_hair_binary.copy()

    if not bounded_decontamination:
        # Legacy broad decontamination (marks all non-core as unknown)
        unknown_region = (core_hair == 0).astype(np.uint8)
        inpaint_radius = 7
    else:
        # Targeted fix: restrict inpainting to a bounded ring around the hair boundary
        if ring_radius_px is not None:
            ring_r = max(3, int(round(ring_radius_px * scale_ref)))
        else:
            ring_r = max(6, int(round(18.0 * scale_ref)))
        inpaint_radius = max(3, int(round(7.0 * scale_ref)))
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_r * 2 + 1, ring_r * 2 + 1))
        dilated_hair = cv2.dilate(ref_hair_binary, k_dilate, iterations=1)
        unknown_region = ((dilated_hair > 0) & (core_hair == 0)).astype(np.uint8)

    if np.any(core_hair > 0) and np.any(unknown_region > 0):
        decontaminated_rgb = cv2.inpaint(reference_rgb, unknown_region, inpaint_radius, cv2.INPAINT_TELEA)
    else:
        decontaminated_rgb = reference_rgb.copy()
    ref_hair_rgb = decontaminated_rgb

    # ── Head geometry ──
    target_geom = {"valid": False}
    ref_geom = {"valid": False}
    if target_landmarks.get("detected"):
        target_geom = estimate_head_geometry(target_landmarks["landmarks_2d"])
    if reference_landmarks.get("detected"):
        ref_geom = estimate_head_geometry(reference_landmarks["landmarks_2d"])

    # ── Compute similarity transform ──
    transform_info = {"valid": False, "is_fallback": False}

    if (target_landmarks.get("detected") and reference_landmarks.get("detected")):
        transform_info = compute_similarity_transform(
            src_landmarks_2d=reference_landmarks["landmarks_2d"],
            dst_landmarks_2d=target_landmarks["landmarks_2d"],
            is_src_mirrored=should_mirror,
            anchor_indices=ALIGNMENT_ANCHOR_INDICES,
        )

    if not transform_info["valid"]:
        reason = transform_info.get("rejection_reason", "Landmarks unavailable")
        warnings.append(f"Similarity transform rejected: {reason}. Using fallback.")
        transform_info = compute_fallback_transform(
            target_geometry=target_geom,
            reference_geometry=ref_geom,
            target_hair_mask_np=target_hair_np,
            ref_hair_mask_np=ref_hair_np,
            target_shape=target_shape,
        )
        if not transform_info["valid"]:
            warnings.append(
                f"Fallback transform also failed: "
                f"{transform_info.get('rejection_reason', 'unknown')}"
            )
            # Return empty alignment with just the original target masks
            hard_prot, trans_w = build_facial_protection_mask(
                target_shape, target_landmarks, target_segmentation_map,
                hairline_transition_px,
            )
            masks = build_alignment_masks(
                aligned_hair_alpha=np.zeros((h_t, w_t), dtype=np.float32),
                target_hair_mask_np=target_hair_np,
                hard_protection=hard_prot,
                transition_weight=trans_w,
                target_shape=target_shape,
                cranial_expansion_px=cranial_expansion_px,
                feather_width_px=feather_width_px,
            )
            return {
                "aligned_hair_rgb": np.zeros((h_t, w_t, 3), dtype=np.uint8),
                "aligned_hair_alpha": np.zeros((h_t, w_t), dtype=np.float32),
                "effective_hair_alpha": np.zeros((h_t, w_t), dtype=np.float32),
                "pasted_composite": target_image.convert("RGB"),
                "edge_repair_mask": masks["target_edit_mask"],
                "target_edit_mask": masks["target_edit_mask"],
                "broad_edit_mask": masks["target_edit_mask"],
                "compositing_alpha": masks["compositing_alpha"],
                "hard_protection": hard_prot,
                "decontaminated_reference_rgb": decontaminated_rgb,
                "reference_hair_alpha": ref_hair_alpha,
                "unknown_region_mask": unknown_region,
                "transform_info": transform_info,
                "pose_info": pose_info,
                "previews": {},
                "preview_paths": {},
                "warnings": warnings,
            }

    # ── Warp reference hair into target space ──
    M = transform_info["matrix"]
    warped_rgb, warped_alpha = warp_reference_hair(
        reference_rgb=ref_hair_rgb,
        reference_alpha=ref_hair_alpha,
        transform_matrix=M,
        target_shape=target_shape,
    )

    # ── Build protection and transition masks (2D distance transform, no horizontal seam) ──
    hard_protection, transition_weight = build_facial_protection_mask(
        target_shape=target_shape,
        target_landmarks=target_landmarks,
        target_segmentation_map=target_segmentation_map,
        hairline_transition_px=hairline_transition_px,
    )

    # ── Harmonize hair color & lighting with restrained tone transfer ──
    harmonized_rgb = harmonize_hair_color(
        aligned_hair_rgb=warped_rgb,
        aligned_hair_alpha=warped_alpha,
        target_rgb=target_rgb,
        target_hair_mask_np=target_hair_np,
        strength=0.40,
    )

    # ── Composite aligned reference hair directly onto target portrait ──
    # Facial features are strictly protected; central curls stay 100% intact
    pasted_composite_np, effective_alpha = composite_aligned_hair(
        target_rgb=target_rgb,
        aligned_hair_rgb=harmonized_rgb,
        aligned_hair_alpha=warped_alpha * transition_weight,
        hard_protection=hard_protection,
    )

    # ── Build dedicated narrow edge repair mask ──
    edge_repair_mask_pil, edge_repair_mask_np = build_edge_repair_mask(
        aligned_hair_alpha=effective_alpha,
        target_hair_mask_np=target_hair_np,
        hard_protection=hard_protection,
        target_shape=target_shape,
        edge_width_px=24,
    )

    # ── Build broad envelope mask and compositing alpha for audit ──
    masks = build_alignment_masks(
        aligned_hair_alpha=warped_alpha,
        target_hair_mask_np=target_hair_np,
        hard_protection=hard_protection,
        transition_weight=transition_weight,
        target_shape=target_shape,
        cranial_expansion_px=cranial_expansion_px,
        feather_width_px=feather_width_px,
    )

    # ── Diagnostic previews ──
    previews = create_alignment_previews(
        target_rgb=target_rgb,
        aligned_hair_rgb=harmonized_rgb,
        aligned_hair_alpha=effective_alpha,
        target_edit_mask_np=edge_repair_mask_np,
        target_landmarks=target_landmarks,
        hard_protection=hard_protection,
        pasted_composite=pasted_composite_np,
        edge_repair_mask_np=edge_repair_mask_np,
    )

    preview_paths = {}
    if save_previews_dir:
        preview_paths = save_alignment_previews(previews, save_previews_dir)
        p_dir = Path(save_previews_dir)
        p_dir.mkdir(parents=True, exist_ok=True)
        try:
            Image.fromarray(decontaminated_rgb).save(p_dir / "decontaminated_reference.png")
            Image.fromarray((ref_hair_alpha * 255).astype(np.uint8)).save(p_dir / "reference_hair_alpha.png")
            Image.fromarray(unknown_region * 255).save(p_dir / "unknown_region_mask.png")
            preview_paths["decontaminated_reference"] = str(p_dir / "decontaminated_reference.png")
            preview_paths["reference_hair_alpha"] = str(p_dir / "reference_hair_alpha.png")
            preview_paths["unknown_region_mask"] = str(p_dir / "unknown_region_mask.png")
        except Exception as exc:
            warnings.append(f"Could not save decontaminated reference preview: {exc}")

    return {
        "aligned_hair_rgb": harmonized_rgb,
        "aligned_hair_alpha": warped_alpha,
        "effective_hair_alpha": effective_alpha,
        "pasted_composite": Image.fromarray(pasted_composite_np),
        "edge_repair_mask": edge_repair_mask_pil,
        "target_edit_mask": masks["target_edit_mask"],
        "broad_edit_mask": masks["target_edit_mask"],
        "compositing_alpha": masks["compositing_alpha"],
        "hard_protection": hard_protection,
        "decontaminated_reference_rgb": decontaminated_rgb,
        "reference_hair_alpha": ref_hair_alpha,
        "unknown_region_mask": unknown_region,
        "transform_info": transform_info,
        "pose_info": pose_info,
        "previews": previews,
        "preview_paths": preview_paths,
        "warnings": warnings,
    }
