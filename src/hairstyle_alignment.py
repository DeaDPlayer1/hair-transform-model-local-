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
    refine_reference_hair_mask,
    decontaminate_reference_hair_rgb,
    PROTECTED_SEG_CLASSES,
    FACIAL_FEATURE_SEG_CLASSES,
    EAR_SEG_CLASSES,
    CLOTHING_SEG_CLASSES,
)
from src.bald_converter import (
    JAWLINE_CONTOUR_INDICES,
    BROW_ARCH_INDICES,
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


def estimate_yaw_robust_scale(ref_geom: Dict[str, Any], target_geom: Dict[str, Any]) -> Optional[float]:
    """Face height (forehead-to-chin) is a vertical measurement and stays
    roughly stable under yaw (left-right head turn), unlike horizontal
    measurements (temple width, interocular distance) which foreshorten as
    the head turns away from frontal. Use it to sanity-check scale when the
    reference and target aren't posed the same way."""
    if not (ref_geom.get("valid") and target_geom.get("valid")):
        return None
    ref_h = ref_geom.get("head_height", 0)
    tgt_h = target_geom.get("head_height", 0)
    if ref_h < 10 or tgt_h < 10:
        return None
    return float(tgt_h / ref_h)


def selective_fill_reference_mask_holes(
    ref_hair_binary: np.ndarray,
    reference_rgb: np.ndarray,
    reference_landmarks: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Selectively fill internal holes (like forehead skin gaps and segmentation dropout)
    while strictly preserving legitimate open spaces between external curls, fine strands,
    and background voids.

    Args:
        ref_hair_binary: (H, W) uint8 binary mask (0 or 255)
        reference_rgb: (H, W, 3) uint8 reference RGB image
        reference_landmarks: optional detected landmarks dict

    Returns:
        (filled_binary, fill_mask, info_dict)
    """
    h, w = ref_hair_binary.shape[:2]
    binary = (ref_hair_binary > 0).astype(np.uint8) * 255
    if not np.any(binary > 0):
        return binary.copy(), np.zeros_like(binary), {"filled_count": 0, "preserved_count": 0, "filled_pixels": 0}

    # Estimate background color from outer border margins
    m = max(5, int(min(h, w) * 0.03))
    corners = np.vstack([
        reference_rgb[:m, :m],
        reference_rgb[:m, -m:],
    ]).reshape(-1, 3)
    bg_mean = corners.mean(axis=0)

    # Scale-aware area thresholds
    total_px = h * w
    tiny_thresh = max(15, int(round(total_px * 0.0001))) # ~100px on 1024x1024
    max_thresh = max(1000, int(round(total_px * 0.004)))  # ~4000px on 1024x1024

    inv = cv2.bitwise_not(binary)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)

    hair_pixels = reference_rgb[binary > 0]
    hair_med = np.median(hair_pixels, axis=0) if len(hair_pixels) > 0 else np.array([45.0, 30.0, 20.0], dtype=np.float32)
    ycrcb = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2YCrCb)

    fill_mask = np.zeros_like(binary)
    filled_components = []
    preserved_components = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Ignore outer background connected to image boundaries
        if x <= 0 or y <= 0 or (x + cw) >= w or (y + ch) >= h:
            continue

        mask_c = (labels == i)
        pix = reference_rgb[mask_c]
        p_mean = pix.mean(axis=0)
        cr_mean = float(np.mean(ycrcb[mask_c, 1]))
        cb_mean = float(np.mean(ycrcb[mask_c, 2]))

        dist_bg = float(np.linalg.norm(p_mean - bg_mean))
        dist_hair = float(np.linalg.norm(p_mean - hair_med))

        is_skin = bool(
            ((p_mean[0] > p_mean[1] > p_mean[2]) and (p_mean[0] - p_mean[2] > 12))
            or (130 <= cr_mean <= 185 and 75 <= cb_mean <= 135 and p_mean[0] > p_mean[2])
        )
        is_bg = bool(
            (dist_bg < 50.0 and dist_bg < dist_hair * 0.9)
            or (p_mean.mean() > 130 and (np.max(p_mean) - np.min(p_mean)) < 35 and dist_hair > 30.0)
        )

        # Decision rules:
        # Enclosed voids showing skin (forehead, temples) or background (curl loops)
        # are legitimate open apertures that MUST NOT be filled into the hair mask.
        # When preserved with alpha = 0, the target's own skin/background shows through.
        # Only tiny segmentation dropouts (<= tiny_thresh) that match interior hair color
        # and are neither skin nor backdrop are filled.
        if is_skin or is_bg:
            preserved_components.append((i, area, "skin_gap" if is_skin else "bg_aperture"))
        elif area <= tiny_thresh and dist_hair < 65.0:
            fill_mask[mask_c] = 255
            filled_components.append((i, area, "tiny_hair_dropout"))
        else:
            preserved_components.append((i, area, "interior_void"))

    filled_binary = cv2.bitwise_or(binary, fill_mask)
    info = {
        "filled_count": len(filled_components),
        "preserved_count": len(preserved_components),
        "filled_pixels": int(np.count_nonzero(fill_mask)),
        "filled_details": filled_components,
    }
    return filled_binary, fill_mask, info


def compute_similarity_transform(
    src_landmarks_2d: np.ndarray,
    dst_landmarks_2d: np.ndarray,
    is_src_mirrored: bool = False,
    anchor_indices: list = ALIGNMENT_ANCHOR_INDICES,
) -> Dict[str, Any]:
    """
    Compute a validated similarity transform (scale + rotation + translation)
    mapping source (reference) landmarks to destination (target) landmarks.

    Uses cv2.estimateAffinePartial2D with scale-independent RANSAC threshold
    normalized by target Interocular Distance (IOD) for robustness across
    arbitrary portrait resolutions.

    Returns:
        dict with:
            - 'valid': bool
            - 'matrix': 2×3 np.ndarray (or None)
            - 'scale': float
            - 'rotation_deg': float
            - 'inliers': int
            - 'confidence': float (inlier ratio)
            - 'rejection_reason': str or None
            - 'details': dict with IOD, NME, residuals
    """
    result = {
        "valid": False,
        "matrix": None,
        "scale": 1.0,
        "rotation_deg": 0.0,
        "inliers": 0,
        "confidence": 0.0,
        "rejection_reason": None,
        "details": {},
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

    # Scale-independent RANSAC threshold based on target Interocular Distance (IOD)
    if len(dst_landmarks_2d) > 263:
        dst_iod = float(np.linalg.norm(dst_landmarks_2d[33] - dst_landmarks_2d[263]))
    else:
        dst_iod = float(np.linalg.norm(dst_pts.max(axis=0) - dst_pts.min(axis=0)) * 0.5)
    dst_iod = max(20.0, dst_iod)

    # Scale-independent RANSAC threshold: 6% of target IOD (e.g. ~31.7px on 2048x2048, ~8px on 512x512)
    reproj_thresh = max(8.0, 0.06 * dst_iod)

    # Estimate similarity transform with RANSAC
    M, inlier_mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_thresh,
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

    # Normalized Mean Reprojection Error (NME)
    ones = np.ones((len(src_pts), 1), dtype=np.float32)
    src_homo = np.hstack([src_pts, ones])
    reprojected = src_homo @ M.T
    residuals = np.linalg.norm(reprojected - dst_pts, axis=1)
    inlier_residuals = residuals[inlier_mask.ravel() > 0]
    mean_residual = float(np.mean(inlier_residuals)) if len(inlier_residuals) > 0 else 999.0
    nme = mean_residual / dst_iod

    result["mode"] = "similarity"
    result["matrix"] = M
    result["scale"] = scale
    result["rotation_deg"] = rotation_deg
    result["tx"] = tx
    result["ty"] = ty
    result["inliers"] = inliers
    result["confidence"] = confidence
    result["details"] = {
        "dst_iod": dst_iod,
        "reproj_thresh_px": reproj_thresh,
        "mean_inlier_residual_px": mean_residual,
        "nme": nme,
        "total_points": len(src_pts),
    }

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

    if nme > 0.15:
        result["rejection_reason"] = (
            f"Excessive normalized reprojection error: {nme:.3f} > 0.15"
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


def extend_hair_to_ear_line(
    aligned_alpha: np.ndarray,
    target_landmarks: Optional[Dict[str, Any]],
    target_shape: Tuple[int, int],
    aligned_rgb: Optional[np.ndarray] = None,
) -> Any:
    """If donor coverage doesn't reach the natural ear-line, locally extend
    it toward the ear instead of rescaling the whole hairstyle. Self-gating:
    styles that already cover this area (curly, voluminous) are untouched
    because their coverage already exceeds the threshold below."""
    h, w = target_shape[:2]
    if not target_landmarks or not target_landmarks.get("detected"):
        return (aligned_alpha, aligned_rgb) if aligned_rgb is not None else aligned_alpha
    lms = target_landmarks["landmarks_2d"]
    iod = float(np.linalg.norm(lms[33] - lms[263])) if len(lms) > 263 else float(w * 0.25)
    half_band = max(4, int(round(iod * 0.10)))

    for ear_idx in (234, 454):
        if ear_idx >= len(lms):
            continue
        ex, ey = lms[ear_idx]
        band_y_top = max(0, int(ey - h * 0.10))
        band_y_bot = min(h, int(ey + h * 0.02))
        x1, x2 = max(0, int(ex) - half_band), min(w, int(ex) + half_band)
        
        # Check coverage specifically at the ear line (lower half of band)
        lower_y = max(0, int(ey - h * 0.04))
        lower_col = aligned_alpha[lower_y:band_y_bot, x1:x2]
        if lower_col.size == 0 or float(np.mean(lower_col > 0.3)) >= 0.65:
            continue  # already covered at ear line -- skip

        src_row = next(
            (y for y in range(band_y_top, band_y_bot)
             if np.mean(aligned_alpha[y, x1:x2] > 0.3) > 0.4),
            None,
        )
        if src_row is None:
            continue
        fill_alpha = aligned_alpha[src_row:src_row + 1, x1:x2]
        fill_rgb = aligned_rgb[src_row:src_row + 1, x1:x2] if aligned_rgb is not None else None

        for y in range(src_row, band_y_bot):
            blend = (y - src_row) / max(1, band_y_bot - src_row)
            new_alpha = np.maximum(aligned_alpha[y, x1:x2], fill_alpha[0] * (1.0 - 0.5 * blend))
            if aligned_rgb is not None and fill_rgb is not None:
                increased = (new_alpha > aligned_alpha[y, x1:x2]) & (new_alpha > 0.1)
                if np.any(increased):
                    aligned_rgb[y, x1:x2][increased] = fill_rgb[0][increased]
            aligned_alpha[y, x1:x2] = new_alpha

    return (aligned_alpha, aligned_rgb) if aligned_rgb is not None else aligned_alpha


# ─── Protected Region Mask ────────────────────────────────────────────

class FacialProtectionResult(tuple):
    """
    Two-tuple of (hard_protection, transition_weight) for full backwards compatibility,
    augmented with identity_protection and buffer_protection attributes.
    """
    def __new__(cls, hard_protection: np.ndarray, transition_weight: np.ndarray, identity_protection: Optional[np.ndarray] = None):
        return super().__new__(cls, (hard_protection, transition_weight))

    def __init__(self, hard_protection: np.ndarray, transition_weight: np.ndarray, identity_protection: Optional[np.ndarray] = None):
        self.hard_protection = hard_protection
        self.buffer_protection = hard_protection
        self.transition_weight = transition_weight
        self.identity_protection = identity_protection if identity_protection is not None else hard_protection


def build_facial_protection_mask(
    target_shape: Tuple[int, int],
    target_landmarks: Optional[Dict[str, Any]] = None,
    target_segmentation_map: Optional[np.ndarray] = None,
    hairline_transition_px: int = HAIRLINE_TRANSITION_PX,
) -> FacialProtectionResult:
    """
    Build facial protection masks and a hairline transition zone.

    Separates:
    - identity_protection: tight, undilated segmentation classes (ears, eyes, brows, nose, mouth)
      to guarantee bit-exact identity preservation.
    - buffer_protection: dilated version used to keep diffusion inpainting from bleeding onto skin.

    Returns:
        FacialProtectionResult (unpacks as (hard_protection, transition_weight)):
            hard_protection:  (H, W) uint8, 255 = dilated safety buffer
            transition_weight: (H, W) float32, 0–1 ramp near the hairline
            .identity_protection: (H, W) uint8, 255 = undilated feature pixels
    """
    h, w = target_shape[:2]
    identity_protection = np.zeros((h, w), dtype=np.uint8)
    hard_protection = np.zeros((h, w), dtype=np.uint8)
    face_feat_mask = np.zeros((h, w), dtype=np.uint8)
    ear_protection = np.zeros((h, w), dtype=np.uint8)
    cloth_protection = np.zeros((h, w), dtype=np.uint8)
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

        # Facial features (eyes, eyebrows, nose, mouth)
        face_feat_tight = np.isin(seg, FACIAL_FEATURE_SEG_CLASSES).astype(np.uint8) * 255
        identity_protection = np.maximum(identity_protection, face_feat_tight)
        face_feat = cv2.dilate(face_feat_tight, kernel, iterations=1)
        face_feat_mask = np.maximum(face_feat_mask, face_feat)
        hard_protection = np.maximum(hard_protection, face_feat)

        # Protect genuine ear cartilage 100%
        ear_mask = np.isin(seg, EAR_SEG_CLASSES).astype(np.uint8) * 255
        ear_protection = ear_mask
        identity_protection = np.maximum(identity_protection, ear_mask)
        hard_protection = np.maximum(hard_protection, ear_protection)

        # Hard-protect clothing
        cloth_protection = np.isin(seg, CLOTHING_SEG_CLASSES).astype(np.uint8) * 255
        identity_protection = np.maximum(identity_protection, cloth_protection)
        hard_protection = np.maximum(hard_protection, cloth_protection)

    # Layer 2: Landmark-based protection (facial features)
    if target_landmarks and target_landmarks.get("detected"):
        lms = target_landmarks["landmarks_2d"]
        pts = np.int32(lms)

        left_eye_idx = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145]
        right_eye_idx = [263, 362, 387, 386, 385, 384, 398, 380, 373, 374]
        mouth_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                     308, 185, 40, 39, 37, 0, 267, 269, 270, 409]
        nose_idx = [168, 6, 197, 195, 5, 4, 1, 2, 98, 327]

        lm_hulls = np.zeros((h, w), dtype=np.uint8)
        for idx_group in [left_eye_idx, right_eye_idx, mouth_idx, nose_idx]:
            valid_pts = [pts[i] for i in idx_group if i < len(pts)]
            if len(valid_pts) >= 3:
                hull = cv2.convexHull(np.array(valid_pts))
                cv2.fillConvexPoly(lm_hulls, hull, 255)

        identity_protection = np.maximum(identity_protection, lm_hulls)
        face_feat_mask = np.maximum(face_feat_mask, lm_hulls)

        # Lower face anatomical polygon: protects cheek skin and jawline
        # from tragus 234 down around chin 152 to tragus 454 and across brow arch
        valid_jaw = [pts[i] for i in JAWLINE_CONTOUR_INDICES if i < len(pts)]
        valid_brow = [pts[i] for i in BROW_ARCH_INDICES if i < len(pts)]
        if len(valid_jaw) >= 10 and len(valid_brow) >= 8:
            poly = np.vstack([valid_jaw, valid_brow])
            face_poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(face_poly_mask, [poly], 255)
            # Erode face polygon slightly so hair can naturally frame the temples/cheeks
            iod = float(np.linalg.norm(lms[33] - lms[263])) if len(lms) > 263 else float(w * 0.25)
            erode_px = max(3, int(round(iod * 0.03)))
            if erode_px % 2 == 0:
                erode_px += 1
            k_poly = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
            face_poly_eroded = cv2.erode(face_poly_mask, k_poly, iterations=1)

            # Temple corners: larger erosion restricted to temple band to give voluminous/curly hair room
            temple_erode_px = max(erode_px, int(round(iod * 0.12)))
            if temple_erode_px % 2 == 0:
                temple_erode_px += 1
            k_temple = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (temple_erode_px, temple_erode_px))
            face_poly_temple_eroded = cv2.erode(face_poly_mask, k_temple, iterations=1)

            temple_band = np.zeros((h, w), dtype=np.uint8)
            for idx in (70, 300, 103, 332, 234, 454):  # outer brow + temple + tragus
                if idx < len(lms):
                    pt_coords = (int(lms[idx][0]), int(lms[idx][1]))
                    cv2.circle(temple_band, pt_coords, int(iod * 0.35), 255, -1)

            face_poly_eroded = np.where(temple_band > 0, face_poly_temple_eroded, face_poly_eroded).astype(np.uint8)

            identity_protection = np.maximum(identity_protection, face_poly_eroded)
            face_feat_mask = np.maximum(face_feat_mask, face_poly_eroded)

        # Dilate landmark / facial protections only — do not expand tight ear protection
        face_feat_mask = cv2.dilate(face_feat_mask, kernel, iterations=1)
        hard_protection = np.maximum(face_feat_mask, ear_protection)
        hard_protection = np.maximum(hard_protection, cloth_protection)

    if target_segmentation_map is not None:
        # Strictly guarantee background wall pixels (seg == 0) are never in protection
        hard_protection[seg == 0] = 0
        identity_protection[seg == 0] = 0

    # ── Scalp Hairline transition zone (smooth 2D distance transform) ──
    # Restrict hairline transition strictly to facial features (brows, eyes, forehead).
    # Ears and clothing maintain full hair opacity (transition_weight = 1.0)
    # so hair naturally frames the ears without creating an artificial hollow gap.
    if np.any(face_feat_mask > 0):
        non_face = (face_feat_mask == 0).astype(np.uint8)
        dist = cv2.distanceTransform(non_face, cv2.DIST_L2, 5)
        trans_px = max(4.0, float(hairline_transition_px))
        transition_weight = np.clip(dist / trans_px, 0.0, 1.0).astype(np.float32)
    elif np.any(hard_protection > 0):
        non_protected = (hard_protection == 0).astype(np.uint8)
        dist = cv2.distanceTransform(non_protected, cv2.DIST_L2, 5)
        trans_px = max(4.0, float(hairline_transition_px))
        transition_weight = np.clip(dist / trans_px, 0.0, 1.0).astype(np.float32)

    return FacialProtectionResult(hard_protection, transition_weight, identity_protection)


# ─── Three-Mask Construction ─────────────────────────────────────────

def build_alignment_masks(
    aligned_hair_alpha: np.ndarray,
    target_hair_mask_np: np.ndarray,
    hard_protection: np.ndarray,
    transition_weight: np.ndarray,
    target_shape: Tuple[int, int],
    cranial_expansion_px: int = 20,
    feather_width_px: float = 6.0,
    donor_coverage_threshold: float = 0.85,
    identity_protection: Optional[np.ndarray] = None,
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
        donor_coverage_threshold: alpha threshold where donor hair fully covers target hair
        identity_protection: optional (H, W) uint8 — tight undilated identity protection

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

    # Binary aligned region (capture entire donor hair footprint)
    aligned_binary = (aligned_clipped > 0.15).astype(np.uint8) * 255

    # Union: old target hair ∪ aligned reference hair
    target_hair_binary = (target_hair_mask_np > 128).astype(np.uint8) * 255
    union_mask = np.maximum(target_hair_binary, aligned_binary)

    # Scale-aware cranial expansion for new hairstyle volume
    if cranial_expansion_px > 0:
        scale_factor = max(0.5, min(h, w) / 512.0)
        k = max(3, int(round(cranial_expansion_px * scale_factor)))
        if k % 2 == 0:
            k += 1
        expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        union_mask = cv2.dilate(union_mask, expand_kernel, iterations=1)

    # Subtract protected regions (tight identity protection if provided, else hard_protection)
    sub_prot = identity_protection if identity_protection is not None else hard_protection
    if sub_prot is not None:
        if sub_prot.shape[:2] != (h, w):
            sub_prot = cv2.resize(sub_prot, (w, h), interpolation=cv2.INTER_NEAREST)
        union_mask[sub_prot > 0] = 0

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
    target_segmentation_map: Optional[np.ndarray] = None,
    target_landmarks: Optional[Dict[str, Any]] = None,
    match_chroma: bool = False,
) -> np.ndarray:
    """
    Match color and lighting temperature of aligned reference hair to target in CIE LAB space.
    Restrained matching preserves donor hair curl texture, highlights, and fine strands while
    harmonizing overall lightness and tone with the target portrait lighting.
    """
    core_indices = aligned_hair_alpha > 0.65
    if not np.any(core_indices):
        return aligned_hair_rgb.copy()

    # Convert RGB to LAB
    ref_bgr = cv2.cvtColor(aligned_hair_rgb, cv2.COLOR_RGB2BGR)
    tgt_bgr = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)

    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    tgt_hair_indices = target_hair_mask_np > 128
    has_target_hair = np.count_nonzero(tgt_hair_indices) > 50

    if has_target_hair:
        tgt_sample = tgt_lab[tgt_hair_indices]
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
    else:
        # Target has no/minimal hair (e.g. Stage 2 bald base portrait).
        # Harmonize lighting exposure and color temperature from target facial/scalp skin,
        # preserving the hair's authentic tonal curve, highlights, and deep shadows.
        skin_sample = None
        if target_segmentation_map is not None:
            # Face and scalp skin (Class 1)
            skin_mask = (target_segmentation_map == 1)
            if np.count_nonzero(skin_mask) > 100:
                skin_sample = tgt_lab[skin_mask]

        if skin_sample is None:
            # Fallback to central face region from geometry
            h, w = target_rgb.shape[:2]
            skin_sample = tgt_lab[int(h * 0.25):int(h * 0.70), int(w * 0.25):int(w * 0.75)].reshape(-1, 3)

        ref_sample = ref_lab[core_indices]
        result_lab = ref_lab.copy()

        # Target ambient lighting temperature from skin chromaticity
        # Skin chromaticity (a*, b*) reflects the ambient lighting warmth/coolness.
        m_skin_a = float(skin_sample[:, 1].mean())
        m_skin_b = float(skin_sample[:, 2].mean())

        # Ambient chromatic shift (warm vs cool light)
        delta_a = float(np.clip(m_skin_a - 14.0, -10.0, 10.0) * 0.25 * strength)
        delta_b = float(np.clip(m_skin_b - 16.0, -12.0, 12.0) * 0.25 * strength)

        # Ambient exposure shift (adjust L gently without crushing contrast or blowing highlights)
        skin_l = float(skin_sample[:, 0].mean())
        l_offset = float(np.clip((skin_l - 62.0) * 0.20 * strength, -15.0, 15.0))

        result_lab[core_indices, 0] = np.clip(ref_lab[core_indices, 0] + l_offset, 0, 255)
        if match_chroma:
            result_lab[core_indices, 1] = np.clip(ref_lab[core_indices, 1] + delta_a, 0, 255)
            result_lab[core_indices, 2] = np.clip(ref_lab[core_indices, 2] + delta_b, 0, 255)

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


class CompositeResult(tuple):
    """
    Two-tuple of (pasted_composite_rgb, effective_alpha) for full backwards compatibility,
    augmented with shadow_map attribute.
    """
    def __new__(cls, pasted_composite_rgb: np.ndarray, effective_alpha: np.ndarray, shadow_map: Optional[np.ndarray] = None):
        return super().__new__(cls, (pasted_composite_rgb, effective_alpha))

    def __init__(self, pasted_composite_rgb: np.ndarray, effective_alpha: np.ndarray, shadow_map: Optional[np.ndarray] = None):
        self.pasted_composite_rgb = pasted_composite_rgb
        self.effective_alpha = effective_alpha
        self.shadow_map = shadow_map


def apply_contact_shadow(
    target_rgb: np.ndarray,
    hair_alpha: np.ndarray,
    target_segmentation_map: Optional[np.ndarray] = None,
    target_landmarks: Optional[Dict[str, Any]] = None,
    identity_protection: Optional[np.ndarray] = None,
    shadow_strength: float = 0.25,
    blur_sigma_scale: float = 1.0,
    vertical_offset_px: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply subtle physical ambient contact occlusion shadow directly beneath overlapping donor hair onto forehead skin.

    Guarantees:
    - Shadow is a separate compositing layer derived from donor alpha.
    - Strictly clipped to visible forehead/temple skin (Class 1).
    - Never adds fake shadow on empty background wall, eyes, eyebrows, or non-overlapping hair.
    - Offset and blur scale proportionally with image resolution.
    - Multiplicative darkening is subtle (default 25% max darkening).

    Returns:
        (shadowed_target_rgb, shadow_map):
            shadowed_target_rgb: (H, W, 3) uint8 image
            shadow_map: (H, W) float32 in [0, 1]
    """
    h, w = target_rgb.shape[:2]
    if shadow_strength <= 0.0:
        return target_rgb.copy(), np.zeros((h, w), dtype=np.float32)

    alpha = hair_alpha
    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    scale = max(h, w) / 512.0
    if vertical_offset_px is None:
        offset_y = max(1, int(round(2.5 * scale)))
    else:
        offset_y = max(0, int(round(vertical_offset_px * scale)))

    if offset_y > 0 and offset_y < h:
        shifted_alpha = np.zeros_like(alpha)
        shifted_alpha[offset_y:, :] = alpha[:-offset_y, :]
    else:
        shifted_alpha = alpha.copy()

    # Narrow contact boundary: positive only immediately beneath terminating hair strands
    shadow_raw = shifted_alpha * np.clip(1.0 - alpha, 0.0, 1.0)

    # Interior low-alpha gaps within hair footprint receive milder occlusion
    gap_k = max(5, int(round(9 * scale)))
    if gap_k % 2 == 0:
        gap_k += 1
    interior_gap = (alpha < 0.25) & (cv2.dilate((alpha > 0.5).astype(np.uint8), np.ones((gap_k, gap_k), np.uint8)) > 0)
    shadow_raw = np.maximum(shadow_raw, interior_gap.astype(np.float32) * 0.5)

    # Gaussian blur for soft natural falloff
    k_size = max(3, int(round(5 * scale * blur_sigma_scale)))
    if k_size % 2 == 0:
        k_size += 1
    sigma = max(0.8, 1.2 * scale * blur_sigma_scale)
    shadow_blur = cv2.GaussianBlur(shadow_raw, (k_size, k_size), sigmaX=sigma, sigmaY=sigma)

    # Strictly clip to visible forehead / temple skin
    skin_mask = np.zeros((h, w), dtype=bool)
    if target_segmentation_map is not None:
        seg = target_segmentation_map
        if seg.shape[:2] != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        skin_mask = (seg == 1)  # Face & scalp skin
    elif target_landmarks and target_landmarks.get("detected"):
        lms = target_landmarks["landmarks_2d"]
        valid_jaw = [lms[i] for i in JAWLINE_CONTOUR_INDICES if i < len(lms)]
        valid_brow = [lms[i] for i in BROW_ARCH_INDICES if i < len(lms)]
        if len(valid_jaw) >= 10 and len(valid_brow) >= 8:
            poly = np.vstack([valid_jaw, valid_brow]).astype(np.int32)
            cv2_skin = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(cv2_skin, [poly], 255)
            skin_mask = cv2_skin > 0
        else:
            skin_mask = np.ones((h, w), dtype=bool)
    else:
        skin_mask = np.ones((h, w), dtype=bool)

    # Eyebrow / eye protection: do not allow shadow below the eyebrow arch
    if target_landmarks and target_landmarks.get("detected"):
        lms = target_landmarks["landmarks_2d"]
        brow_indices = [70, 105, 63, 300, 334, 296]
        valid_b = [lms[i][1] for i in brow_indices if i < len(lms)]
        if valid_b:
            brow_y_limit = float(np.min(valid_b)) + max(2.0, 4.0 * scale)
            skin_mask[int(brow_y_limit):, :] = False

    if identity_protection is not None:
        ident = identity_protection
        if ident.shape[:2] != (h, w):
            ident = cv2.resize(ident, (w, h), interpolation=cv2.INTER_NEAREST)
        skin_mask[ident > 0] = False

    if target_segmentation_map is not None:
        # Zero out background wall
        skin_mask[seg == 0] = False

    # Check if hair overlaps forehead skin:
    # Short crops or styles without bangs that do not overlap the forehead should not receive shadow
    hair_on_forehead = (alpha > 0.3) & skin_mask
    min_overlap_px = int(round(50 * (scale ** 2)))
    if np.count_nonzero(hair_on_forehead) < min_overlap_px:
        return target_rgb.copy(), np.zeros((h, w), dtype=np.float32)

    shadow_map = np.zeros((h, w), dtype=np.float32)
    shadow_map[skin_mask] = shadow_blur[skin_mask]
    shadow_map = np.clip(shadow_map * float(np.clip(shadow_strength, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)

    # Ambient occlusion darkening
    shadow_factor = (1.0 - shadow_map)[..., None]
    shadowed_target = np.clip(target_rgb.astype(np.float32) * shadow_factor, 0, 255).astype(np.uint8)
    return shadowed_target, shadow_map


def composite_aligned_hair(
    target_rgb: np.ndarray,
    aligned_hair_rgb: np.ndarray,
    aligned_hair_alpha: np.ndarray,
    hard_protection: Optional[np.ndarray] = None,
    soft_transition_px: float = 3.0,
    ear_mask: Optional[np.ndarray] = None,
    contact_shadow_strength: float = 0.25,
    target_segmentation_map: Optional[np.ndarray] = None,
    target_landmarks: Optional[Dict[str, Any]] = None,
    identity_protection: Optional[np.ndarray] = None,
) -> CompositeResult:
    """
    Composite aligned reference hair onto target image with facial feature protection
    and physical contact occlusion shadowing beneath overlapping hair strands.

    Returns:
        CompositeResult (unpacks as (pasted_composite_rgb, effective_alpha)):
            pasted_composite_rgb: (H, W, 3) uint8 composite image
            effective_alpha: (H, W) float32 in [0, 1]
            .shadow_map: (H, W) float32 in [0, 1] contact shadow map
    """
    h, w = target_rgb.shape[:2]
    hair_rgb = aligned_hair_rgb
    hair_alpha = aligned_hair_alpha

    if hair_rgb.shape[:2] != (h, w):
        hair_rgb = cv2.resize(hair_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    if hair_alpha.shape[:2] != (h, w):
        hair_alpha = cv2.resize(hair_alpha, (w, h), interpolation=cv2.INTER_LINEAR)

    alpha = np.clip(hair_alpha, 0.0, 1.0).astype(np.float32)

    # 1. Apply ambient contact shadow onto skin beneath overlapping strands
    if contact_shadow_strength > 0.0:
        target_base, shadow_map = apply_contact_shadow(
            target_rgb=target_rgb,
            hair_alpha=alpha,
            target_segmentation_map=target_segmentation_map,
            target_landmarks=target_landmarks,
            identity_protection=identity_protection if identity_protection is not None else hard_protection,
            shadow_strength=contact_shadow_strength,
        )
    else:
        target_base = target_rgb.copy()
        shadow_map = np.zeros((h, w), dtype=np.float32)

    if hard_protection is not None:
        if hard_protection.shape[:2] != (h, w):
            hard_protection = cv2.resize(hard_protection, (w, h), interpolation=cv2.INTER_NEAREST)

        # When ear_mask is present, exclude ears from soft dissolve so genuine curls
        # naturally frame and fall over/behind the ears without artificial gaps.
        face_hard = hard_protection.copy()
        if ear_mask is not None and np.any(ear_mask > 0):
            if ear_mask.shape[:2] != (h, w):
                ear_mask = cv2.resize(ear_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            face_hard[ear_mask > 0] = 0

        # Fractional smoothly-weighted transition in narrow adjacent hair rim
        if soft_transition_px > 0.0 and np.any(face_hard > 0):
            dist_outside = cv2.distanceTransform((face_hard == 0).astype(np.uint8), cv2.DIST_L2, 5)
            smooth_weight = np.clip(dist_outside / float(soft_transition_px), 0.0, 1.0).astype(np.float32)
            alpha = alpha * smooth_weight

        # Strictly zero out alpha inside face_hard (100% bit-exact facial protection)
        alpha[face_hard > 0] = 0.0

    alpha_3ch = alpha[..., None]
    composite = (
        target_base.astype(np.float32) * (1.0 - alpha_3ch) +
        hair_rgb.astype(np.float32) * alpha_3ch
    ).clip(0, 255).astype(np.uint8)

    return CompositeResult(composite, alpha, shadow_map)


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill fully-enclosed holes in a binary mask (uint8, values 0 or 255).
    Uses OpenCV floodFill without requiring external dependencies.
    """
    if not np.any(mask > 0):
        return mask.copy()
    h, w = mask.shape[:2]
    padded = cv2.copyMakeBorder(mask, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ff_mask = np.zeros((h + 6, w + 6), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, holes)
    return filled[2 : 2 + h, 2 : 2 + w]


def build_edge_repair_mask(
    aligned_hair_alpha: np.ndarray,
    target_hair_mask_np: np.ndarray,
    hard_protection: Optional[np.ndarray],
    target_shape: Tuple[int, int],
    edge_width_px: int = 12,
    ear_mask_np: Optional[np.ndarray] = None,
    donor_coverage_threshold: float = 0.85,
    identity_protection: Optional[np.ndarray] = None,
    buffer_protection: Optional[np.ndarray] = None,
    target_segmentation_map: Optional[np.ndarray] = None,
    target_landmarks: Optional[Dict[str, Any]] = None,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Build a dedicated, narrow edge-repair mask around the internal hairline seam.

    Guarantees:
    1. The central hair texture and curl pattern from the reference composite are EXCLUDED
       from inpainting (their mask value is 0), so diffusion does NOT repaint them.
    2. Outer curls and hair silhouettes against empty background wall are STRICTLY EXCLUDED (0)
       (preserving 100% authentic photographic curls and eliminating glassy/bubbly rings).
    3. The repair ribbon is strictly bounded to the internal intersection between donor hair
       roots and the target head/scalp/forehead interface (hairline seam, temples, sideburns).
    4. Uncovered old hair removal is enabled ONLY when genuine residual hair (> 500 px) is
       present in target_hair_mask_np, preventing hallucinated inpainting on clean bald bases.
    5. Identity Protection: Genuine ear cartilage, facial features (eyes, eyebrows, nose, mouth),
       and clothing are STRICTLY and UNCONDITIONALLY EXCLUDED (0) under all inputs.
    6. Buffer Protection: Dilated feature/ear margins block non-hair inpainting bleed onto skin.

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

    # A. Determine genuine target hair and permitted interface region
    residual_hair_count = int(np.count_nonzero(tgt_mask > 128))
    has_genuine_target_hair = (residual_hair_count > 500) or (target_segmentation_map is None and residual_hair_count > 0)

    # Permitted interface: only regions where donor hair interfaces with target head / scalp / skin / neck
    if target_segmentation_map is not None:
        seg = target_segmentation_map
        if seg.shape[:2] != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        # Person classes: face skin (1), facial features (2-13), neck (14), neck_l (15), cloth (16)
        target_person_mask = np.isin(seg, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).astype(np.uint8) * 255
        if has_genuine_target_hair:
            target_person_mask = np.maximum(target_person_mask, (tgt_mask > 128).astype(np.uint8) * 255)
        permitted_interface = target_person_mask.copy()
        # Strictly zero out background wall (seg == 0) so outer curls against background remain untouched
        permitted_interface[seg == 0] = 0
    elif hard_protection is not None and np.any(hard_protection > 0):
        target_head = np.maximum((tgt_mask > 128).astype(np.uint8) * 255, hard_protection)
        permitted_interface = cv2.dilate(target_head, k_band, iterations=2)
    else:
        # Fallback for synthetic unit tests with no person or landmarks provided
        permitted_interface = None

    # B. Hair silhouette boundaries
    hair_core = (alpha >= donor_coverage_threshold).astype(np.uint8) * 255
    hair_footprint = (alpha > 0.15).astype(np.uint8) * 255

    # Boundary band via morphological gradient
    dilated_footprint = cv2.dilate(hair_footprint, k_band, iterations=1)
    eroded_core = cv2.erode(hair_core, k_band, iterations=1)
    boundary_band = cv2.subtract(dilated_footprint, eroded_core)

    # Strictly confine boundary band to permitted head interface
    # This completely eliminates outer silhouette curls in empty background wall
    if permitted_interface is not None:
        boundary_band = cv2.bitwise_and(boundary_band, permitted_interface)

    # Morphological CLOSE pass to merge fragmented micro-islands on curly/thin hair
    k_close_size = max(5, int(round(w_px * 0.4)))
    if k_close_size % 2 == 0:
        k_close_size += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close_size, k_close_size))
    boundary_band = cv2.morphologyEx(boundary_band, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    boundary_band = _fill_mask_holes(boundary_band)
    if permitted_interface is not None:
        boundary_band = cv2.bitwise_and(boundary_band, permitted_interface)

    # C. Uncovered old target hair: enabled ONLY when genuine residual hair was detected
    if has_genuine_target_hair:
        old_hair = (tgt_mask > 128).astype(np.uint8) * 255
        covered_by_new = (alpha >= donor_coverage_threshold).astype(np.uint8) * 255
        uncovered_old_hair = cv2.subtract(old_hair, covered_by_new)
        if np.any(uncovered_old_hair > 0):
            k_old_size = max(3, int(round(w_px * 0.4)))
            if k_old_size % 2 == 0:
                k_old_size += 1
            k_old = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_old_size, k_old_size))
            uncovered_old_hair_expanded = cv2.dilate(uncovered_old_hair, k_old, iterations=1)
            if permitted_interface is not None:
                uncovered_old_hair_expanded = cv2.bitwise_and(uncovered_old_hair_expanded, permitted_interface)
        else:
            uncovered_old_hair_expanded = np.zeros((h, w), dtype=np.uint8)
    else:
        uncovered_old_hair = np.zeros((h, w), dtype=np.uint8)
        uncovered_old_hair_expanded = np.zeros((h, w), dtype=np.uint8)

    # D. Combine boundary seam + uncovered old hair
    combined_mask = np.maximum(boundary_band, uncovered_old_hair_expanded)

    # E. Strictly protect central hair core (keep reference curls intact)
    central_protected_hair = cv2.subtract(eroded_core, uncovered_old_hair_expanded)
    combined_mask[central_protected_hair > 0] = 0

    # Resolve protection masks
    buf_prot = buffer_protection
    if buf_prot is None and hard_protection is not None:
        buf_prot = hard_protection.copy()
    if buf_prot is not None and buf_prot.shape[:2] != (h, w):
        buf_prot = cv2.resize(buf_prot, (w, h), interpolation=cv2.INTER_NEAREST)

    ident_prot = identity_protection
    if ident_prot is None and hard_protection is not None:
        ident_prot = getattr(hard_protection, "identity_protection", None)
    if ident_prot is not None and ident_prot.shape[:2] != (h, w):
        ident_prot = cv2.resize(ident_prot, (w, h), interpolation=cv2.INTER_NEAREST)

    # Fold ear cartilage into identity_protection and ear buffer into buf_prot
    if ear_mask_np is not None and np.any(ear_mask_np > 0):
        ear_m = ear_mask_np
        if ear_m.shape[:2] != (h, w):
            ear_m = cv2.resize(ear_m, (w, h), interpolation=cv2.INTER_NEAREST)
        ear_cartilage = (ear_m > 0).astype(np.uint8) * 255
        if ident_prot is not None:
            ident_prot = np.maximum(ident_prot, ear_cartilage)
        else:
            ident_prot = ear_cartilage.copy()

        k_buf_size = max(5, int(round(w_px * 1.2)))
        if k_buf_size % 2 == 0:
            k_buf_size += 1
        k_ear_buf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_buf_size, k_buf_size))
        ear_buffer = cv2.dilate(ear_cartilage, k_ear_buf, iterations=1)
        if buf_prot is not None:
            buf_prot = np.maximum(buf_prot, ear_buffer)
        else:
            buf_prot = ear_buffer

    # F. Strictly protect identity (unconditionally zero out over real skin/ear/eye/mouth pixels)
    if ident_prot is not None:
        combined_mask[ident_prot > 0] = 0

    # G. Buffer protection: dilated safety margin around features and ears
    if buf_prot is not None:
        combined_mask[(buf_prot > 0) & (uncovered_old_hair == 0) & (boundary_band == 0)] = 0

    # H. Background wall protection: strictly zero out background wall (seg == 0)
    if target_segmentation_map is not None:
        seg = target_segmentation_map
        if seg.shape[:2] != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        combined_mask[seg == 0] = 0

    edge_mask_np = (combined_mask > 128).astype(np.uint8) * 255
    edge_mask_pil = Image.fromarray(edge_mask_np, mode="L")
    return edge_mask_pil, edge_mask_np


def validate_edge_repair_mask(
    edge_repair_mask_np: np.ndarray,
    aligned_hair_alpha: np.ndarray,
    target_segmentation_map: Optional[np.ndarray] = None,
    identity_protection: Optional[np.ndarray] = None,
    max_area_ratio: float = 0.20,
    max_outer_bg_ratio: float = 0.05,
) -> Dict[str, Any]:
    """
    Validate edge repair mask before Stable Diffusion inpainting.

    Measures:
    - repair_area_pct: fraction of canvas covered by inpainting mask
    - identity_overlap_count: count of protected facial feature pixels inside mask
    - outer_bg_overlap_pct: fraction of mask located out in empty background
    - should_inpaint: whether diffusion should run
    - is_valid: whether the mask satisfies all safety invariants
    """
    h, w = edge_repair_mask_np.shape[:2]
    total_pixels = h * w
    active_mask = (edge_repair_mask_np > 128)
    active_pixels = int(np.count_nonzero(active_mask))
    area_pct = round((active_pixels / total_pixels) * 100.0, 3)

    ident_overlap = 0
    if identity_protection is not None:
        ident_overlap = int(np.count_nonzero(active_mask & (identity_protection > 0)))

    outer_bg_pixels = 0
    outer_bg_pct = 0.0
    if target_segmentation_map is not None:
        bg_mask = (target_segmentation_map == 0)
        outer_bg_pixels = int(np.count_nonzero(active_mask & bg_mask))
        if active_pixels > 0:
            outer_bg_pct = round((outer_bg_pixels / active_pixels) * 100.0, 3)

    is_valid = True
    reasons = []

    if ident_overlap > 0:
        is_valid = False
        reasons.append(f"Identity overlap violation: {ident_overlap} protected pixels in repair mask")

    if (active_pixels / total_pixels) > max_area_ratio:
        is_valid = False
        reasons.append(f"Mask area too large: {area_pct}% exceeds {max_area_ratio*100}% threshold")

    if (outer_bg_pct / 100.0) > max_outer_bg_ratio:
        is_valid = False
        reasons.append(f"Outer background overlap too large: {outer_bg_pct}% in empty background")

    # If active_pixels == 0, pre-diffusion composite is already clean; diffusion is skipped safely
    should_inpaint = is_valid and (active_pixels > 0)

    sanitized_mask_np = edge_repair_mask_np.copy()
    if identity_protection is not None:
        sanitized_mask_np[identity_protection > 0] = 0

    return {
        "is_valid": is_valid,
        "should_inpaint": should_inpaint,
        "active_pixels": active_pixels,
        "area_pct": area_pct,
        "identity_overlap_count": ident_overlap,
        "outer_bg_pixels": outer_bg_pixels,
        "outer_bg_pct": outer_bg_pct,
        "reasons": reasons,
        "sanitized_mask_np": sanitized_mask_np,
    }


def create_repair_mask_validation_overlay(
    target_rgb: np.ndarray,
    edge_repair_mask_np: np.ndarray,
    target_segmentation_map: Optional[np.ndarray] = None,
    identity_protection: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create a 3-channel visual diagnostic overlay of the repair mask validation.
    - Green: Valid repair seam
    - Blue outline: Target person boundary
    - Yellow outline: Protected identity region
    - Red: Mask pixels violating background or identity constraints
    """
    h, w = target_rgb.shape[:2]
    overlay = target_rgb.copy().astype(np.float32)

    active = edge_repair_mask_np > 128
    bg_mask = (target_segmentation_map == 0) if target_segmentation_map is not None else np.zeros((h, w), dtype=bool)
    ident_mask = (identity_protection > 0) if identity_protection is not None else np.zeros((h, w), dtype=bool)

    invalid_mask = active & (bg_mask | ident_mask)
    valid_mask = active & (~invalid_mask)

    # Tint valid repair seam in Green
    overlay[valid_mask] = overlay[valid_mask] * 0.4 + np.array([0, 255, 0], dtype=np.float32) * 0.6
    # Tint invalid mask in Red
    overlay[invalid_mask] = overlay[invalid_mask] * 0.3 + np.array([255, 0, 0], dtype=np.float32) * 0.7

    # Tint identity protection outline in Yellow
    if np.any(ident_mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ident_edge = cv2.subtract(cv2.dilate(ident_mask.astype(np.uint8), k), ident_mask.astype(np.uint8)) > 0
        overlay[ident_edge] = np.array([255, 255, 0], dtype=np.float32)

    return np.clip(overlay, 0, 255).astype(np.uint8)


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
    reference_segmentation_map: Optional[np.ndarray] = None,
    cranial_expansion_px: int = 20,
    feather_width_px: float = 6.0,
    hairline_transition_px: int = HAIRLINE_TRANSITION_PX,
    save_previews_dir: Optional[str] = None,
    bounded_decontamination: bool = True,
    ring_radius_px: Optional[int] = None,
    donor_coverage_threshold: float = 0.85,
    contact_shadow_strength: float = 0.25,
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

    # ── Prune false-positive studio backdrop & exclude donor forehead skin ──
    ref_hair_np_clean, prune_info = refine_reference_hair_mask(
        reference_rgb,
        ref_hair_np,
        reference_landmarks,
        reference_segmentation_map=reference_segmentation_map,
    )
    pruned_count = int(prune_info["pruned_count"])
    if pruned_count > 0:
        warnings.append(f"Pruned {pruned_count} false-positive backdrop pixels from reference hair mask.")
    skin_excl_count = int(prune_info.get("skin_exclusion_count", 0))
    if skin_excl_count > 0:
        warnings.append(f"Excluded {skin_excl_count} donor forehead skin pixels from reference hair mask.")

    # ── Selective mask-hole cleanup ──
    ref_hair_filled, fill_mask, fill_info = selective_fill_reference_mask_holes(
        ref_hair_np_clean, reference_rgb, reference_landmarks
    )
    if fill_info["filled_count"] > 0:
        warnings.append(f"Selectively filled {fill_info['filled_count']} internal mask holes ({fill_info['filled_pixels']} px).")

    # ── Build reference hair alpha (soft-edge) ──
    ref_hair_binary = (ref_hair_filled > 128).astype(np.uint8)
    ref_hair_dist = cv2.distanceTransform(ref_hair_binary, cv2.DIST_L2, 5)
    ref_hair_alpha = np.clip(ref_hair_dist / 3.0, 0.0, 1.0).astype(np.float32)

    # ── Foreground color decontamination (hair-rim only in bounded mode) ──
    decontaminated_rgb, unknown_region = decontaminate_reference_hair_rgb(
        reference_rgb,
        ref_hair_binary,
        bounded_decontamination=bounded_decontamination,
        ring_radius_px=ring_radius_px,
        donor_skin_color=prune_info.get("donor_skin_color"),
    )
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
            prot_res = build_facial_protection_mask(
                target_shape, target_landmarks, target_segmentation_map,
                hairline_transition_px,
            )
            hard_prot, trans_w = prot_res[0], prot_res[1]
            ident_prot = getattr(prot_res, "identity_protection", hard_prot)
            masks = build_alignment_masks(
                aligned_hair_alpha=np.zeros((h_t, w_t), dtype=np.float32),
                target_hair_mask_np=target_hair_np,
                hard_protection=hard_prot,
                transition_weight=trans_w,
                target_shape=target_shape,
                cranial_expansion_px=cranial_expansion_px,
                feather_width_px=feather_width_px,
                identity_protection=ident_prot,
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
                "identity_protection": ident_prot,
                "buffer_protection": hard_prot,
                "decontaminated_reference_rgb": decontaminated_rgb,
                "reference_hair_alpha": ref_hair_alpha,
                "unknown_region_mask": unknown_region,
                "transform_info": transform_info,
                "pose_info": pose_info,
                "previews": {},
                "preview_paths": {},
                "warnings": warnings,
            }

    # ── Pose-mismatch scale sanity guard ──
    # Horizontal-distance-based scale (RANSAC and the fallback both use it)
    # inflates under yaw. Clamp against a yaw-robust vertical estimate.
    yaw_mismatch = pose_info.get("mirrored_diff", 0.0) if should_mirror else pose_info.get("unmirrored_diff", 0.0)

    if transform_info.get("valid") and yaw_mismatch > 20.0:
        plausible_scale = estimate_yaw_robust_scale(ref_geom, target_geom)
        if plausible_scale is not None:
            computed_scale = float(transform_info.get("scale", 1.0))
            ratio = computed_scale / plausible_scale
            if ratio > 1.3 or ratio < 0.7:
                correction = plausible_scale / computed_scale
                M = transform_info["matrix"].copy()
                if target_geom.get("valid") and "nasion" in target_geom:
                    cx = float(target_geom["nasion"][0])
                    cy = float(target_geom["nasion"][1])
                else:
                    cx = float(w_t / 2.0)
                    cy = float(h_t / 2.0)
                M[:, :2] *= correction
                M[0, 2] = cx - correction * (cx - M[0, 2])
                M[1, 2] = cy - correction * (cy - M[1, 2])
                transform_info["matrix"] = M
                transform_info["original_scale"] = computed_scale
                transform_info["scale"] = plausible_scale
                transform_info["scale_corrected"] = True
                transform_info["scale_correction_ratio"] = round(ratio, 3)
                warnings.append(
                    f"Scale correction: {computed_scale:.2f}x deviated {ratio:.2f}x from "
                    f"the yaw-robust estimate ({plausible_scale:.2f}x) under a "
                    f"{yaw_mismatch:.1f}° pose mismatch — rescaled around the target's face center."
                )

        if yaw_mismatch > 45.0:
            transform_info["pose_quality_flag"] = "poor"
            warnings.append(
                f"REFERENCE POSE WARNING: this reference is turned ~{yaw_mismatch:.0f}° "
                "from frontal. A 2D CPU pipeline can't reconstruct the occluded side of "
                "the hairstyle from this angle — the correction above reduces distortion "
                "but can't fully compensate. For best results, use a closer-to-frontal reference."
            )

    # ── Warp reference hair into target space ──
    M = transform_info["matrix"]
    warped_rgb, warped_alpha = warp_reference_hair(
        reference_rgb=ref_hair_rgb,
        reference_alpha=ref_hair_alpha,
        transform_matrix=M,
        target_shape=target_shape,
    )

    # Magnified source texture reads as jagged/blocky — smooth proportionally
    # to how much enlargement actually happened.
    applied_scale = float(transform_info.get("scale", 1.0))
    if applied_scale > 1.3:
        smooth_px = max(1, int(round((applied_scale - 1.0) * 6)))
        k = smooth_px * 2 + 1
        warped_alpha = cv2.GaussianBlur(warped_alpha, (k, k), 0)

    warped_alpha, warped_rgb = extend_hair_to_ear_line(
        warped_alpha, target_landmarks, target_shape, aligned_rgb=warped_rgb
    )

    # ── Build protection and transition masks (2D distance transform, no horizontal seam) ──
    prot_res = build_facial_protection_mask(
        target_shape=target_shape,
        target_landmarks=target_landmarks,
        target_segmentation_map=target_segmentation_map,
        hairline_transition_px=hairline_transition_px,
    )
    hard_protection, transition_weight = prot_res[0], prot_res[1]
    identity_protection = getattr(prot_res, "identity_protection", hard_protection)

    # ── Harmonize hair color & lighting with restrained tone transfer ──
    harmonized_rgb = harmonize_hair_color(
        aligned_hair_rgb=warped_rgb,
        aligned_hair_alpha=warped_alpha,
        target_rgb=target_rgb,
        target_hair_mask_np=target_hair_np,
        strength=0.15,
        target_segmentation_map=target_segmentation_map,
        target_landmarks=target_landmarks,
        match_chroma=False,
    )

    # ── Scale-aware parameter derivation from head geometry ──
    soft_transition_px = 3.0
    if target_geom.get("valid") and target_geom.get("interocular_dist"):
        iod = float(target_geom["interocular_dist"])
        edge_width_px = int(np.clip(round(iod * 0.12), 8, 20))
        feather_width_px = float(np.clip(round(iod * 0.06), 4.0, 10.0))
        soft_transition_px = float(np.clip(round(iod * 0.025), 3.0, 6.0))
        cranial_expansion_px = int(np.clip(round(iod * 0.20), 12, 30))
    else:
        edge_width_px = 12

    ear_mask_np = None
    if target_segmentation_map is not None:
        ear_mask_np = np.isin(target_segmentation_map, EAR_SEG_CLASSES).astype(np.uint8) * 255

    # ── Composite aligned reference hair directly onto target portrait ──
    # Facial features are strictly protected; genuine curls flow naturally around ears;
    # subtle contact shadow physically grounds overlapping bangs/strands onto forehead skin
    comp_res = composite_aligned_hair(
        target_rgb=target_rgb,
        aligned_hair_rgb=harmonized_rgb,
        aligned_hair_alpha=warped_alpha,
        hard_protection=hard_protection,
        soft_transition_px=soft_transition_px,
        ear_mask=ear_mask_np,
        contact_shadow_strength=contact_shadow_strength,
        target_segmentation_map=target_segmentation_map,
        target_landmarks=target_landmarks,
        identity_protection=identity_protection,
    )
    pasted_composite_np, effective_alpha = comp_res
    shadow_map = getattr(comp_res, "shadow_map", np.zeros(target_shape, dtype=np.float32))

    # ── Build dedicated narrow edge repair mask (strictly confined to internal head/hairline seam) ──
    edge_repair_mask_pil, edge_repair_mask_np = build_edge_repair_mask(
        aligned_hair_alpha=effective_alpha,
        target_hair_mask_np=target_hair_np,
        hard_protection=hard_protection,
        target_shape=target_shape,
        edge_width_px=edge_width_px,
        ear_mask_np=ear_mask_np,
        donor_coverage_threshold=donor_coverage_threshold,
        identity_protection=identity_protection,
        buffer_protection=hard_protection,
        target_segmentation_map=target_segmentation_map,
        target_landmarks=target_landmarks,
    )

    # ── Validate edge repair mask before diffusion ──
    validation_res = validate_edge_repair_mask(
        edge_repair_mask_np=edge_repair_mask_np,
        aligned_hair_alpha=effective_alpha,
        target_segmentation_map=target_segmentation_map,
        identity_protection=identity_protection,
    )
    validation_overlay = create_repair_mask_validation_overlay(
        target_rgb=target_rgb,
        edge_repair_mask_np=edge_repair_mask_np,
        target_segmentation_map=target_segmentation_map,
        identity_protection=identity_protection,
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
        donor_coverage_threshold=donor_coverage_threshold,
        identity_protection=identity_protection,
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
    previews["repair_mask_validation_overlay"] = validation_overlay
    if shadow_map is not None and np.any(shadow_map > 0):
        previews["contact_shadow_map"] = (shadow_map * 255).astype(np.uint8)

    ref_skin_excl_img = prune_info.get("reference_skin_exclusion_mask")
    ref_alpha_refined_img = Image.fromarray((ref_hair_alpha * 255).astype(np.uint8))
    ref_mask_sbs_img = prune_info.get("reference_hair_mask_before_after")
    donor_bleed_prev_img = prune_info.get("donor_skin_bleed_preview")

    preview_paths = {}
    if save_previews_dir:
        preview_paths = save_alignment_previews(previews, save_previews_dir)
        p_dir = Path(save_previews_dir)
        p_dir.mkdir(parents=True, exist_ok=True)
        try:
            Image.fromarray(decontaminated_rgb).save(p_dir / "decontaminated_reference.png")
            Image.fromarray((ref_hair_alpha * 255).astype(np.uint8)).save(p_dir / "reference_hair_alpha.png")
            Image.fromarray(unknown_region * 255).save(p_dir / "unknown_region_mask.png")
            Image.fromarray(ref_hair_filled).save(p_dir / "reference_hair_mask_filled.png")
            Image.fromarray(fill_mask).save(p_dir / "reference_hair_mask_holes.png")
            Image.fromarray(validation_overlay).save(p_dir / "repair_mask_validation_overlay.png")
            if ref_skin_excl_img is not None:
                ref_skin_excl_img.save(p_dir / "reference_skin_exclusion_mask.png")
            ref_alpha_refined_img.save(p_dir / "reference_hair_alpha_refined.png")
            if ref_mask_sbs_img is not None:
                ref_mask_sbs_img.save(p_dir / "reference_hair_mask_before_after.png")
            if donor_bleed_prev_img is not None:
                donor_bleed_prev_img.save(p_dir / "donor_skin_bleed_preview.png")
            if shadow_map is not None:
                Image.fromarray((shadow_map * 255).astype(np.uint8)).save(p_dir / "contact_shadow_map.png")
            preview_paths["decontaminated_reference"] = str(p_dir / "decontaminated_reference.png")
            preview_paths["reference_hair_alpha"] = str(p_dir / "reference_hair_alpha.png")
            preview_paths["unknown_region_mask"] = str(p_dir / "unknown_region_mask.png")
            preview_paths["reference_hair_mask_filled"] = str(p_dir / "reference_hair_mask_filled.png")
            preview_paths["reference_hair_mask_holes"] = str(p_dir / "reference_hair_mask_holes.png")
            preview_paths["repair_mask_validation_overlay"] = str(p_dir / "repair_mask_validation_overlay.png")
            if shadow_map is not None:
                preview_paths["contact_shadow_map"] = str(p_dir / "contact_shadow_map.png")
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
        "identity_protection": identity_protection,
        "buffer_protection": hard_protection,
        "decontaminated_reference_rgb": decontaminated_rgb,
        "reference_hair_alpha": ref_hair_alpha,
        "reference_skin_exclusion_mask": ref_skin_excl_img,
        "reference_hair_alpha_refined": ref_alpha_refined_img,
        "reference_hair_mask_before_after": ref_mask_sbs_img,
        "donor_skin_bleed_preview": donor_bleed_prev_img,
        "reference_mask_filled": ref_hair_filled,
        "reference_fill_mask": fill_mask,
        "fill_info": fill_info,
        "unknown_region_mask": unknown_region,
        "transform_info": transform_info,
        "pose_info": pose_info,
        "previews": previews,
        "preview_paths": preview_paths,
        "contact_shadow_map": shadow_map,
        "validation_result": validation_res,
        "validation_overlay": validation_overlay,
        "warnings": warnings,
    }
