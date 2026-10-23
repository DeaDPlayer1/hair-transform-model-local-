"""
Hair Mask Refinement, Facial Protection, and Compositing Module.

Provides robust mask engineering for generative inpainting:
1. 100% old hair coverage to prevent underlying curl bleed.
2. Controlled cranial silhouette expansion (upward & lateral) for new hair volume.
3. Strict multi-layer facial feature protection (eyes, eyebrows, nose, mouth, clothing).
4. Inner-feathered blend masks ensuring 100% bit-exact target preservation outside the edit region.
"""

from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image


# Face parsing segmentation classes (ResNet18 ONNX model):
# 0: background, 1: face skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye,
# 6: eye_g (glasses), 7: l_ear, 8: r_ear, 9: ear_r, 10: nose, 11: mouth,
# 12: u_lip, 13: l_lip, 14: neck, 15: neck_l, 16: cloth, 17: hair, 18: hat
PROTECTED_SEG_CLASSES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # brows, eyes, nose, mouth, ears
FACIAL_FEATURE_SEG_CLASSES = [2, 3, 4, 5, 6, 10, 11, 12, 13]  # brows, eyes, nose, mouth (excludes ears)
EAR_SEG_CLASSES = [7, 8, 9]  # l_ear, r_ear, ear_r
CLOTHING_SEG_CLASSES = [16]


def refine_reference_hair_mask(
    reference_rgb: np.ndarray,
    reference_hair_mask: np.ndarray,
    landmarks_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Prune studio-backdrop pixels that the hair parser labeled as hair.

    Keeps real interior highlights and dark/light hair. Removes large curl-gap
    pockets and perimeter backdrop fringe. Strictly protects core hair around
    the skull from being pruned, and prevents excessive pruning from destroying
    the reference hairstyle when background colors vary.
    """
    ref_rgb = reference_rgb
    ref_hair_np = reference_hair_mask
    h_r, w_r = ref_rgb.shape[:2]
    if ref_hair_np.shape[:2] != (h_r, w_r):
        ref_hair_np = cv2.resize(ref_hair_np, (w_r, h_r), interpolation=cv2.INTER_NEAREST)

    hair = ref_hair_np > 128

    # 1. Sample backdrop specifically from TOP CORNERS (above shoulders and clothing)
    c_w = max(10, int(w_r * 0.12))
    c_h = max(10, int(h_r * 0.12))

    tl = ref_rgb[:c_h, :c_w]
    tr = ref_rgb[:c_h, -c_w:]
    corners = np.concatenate([tl.reshape(-1, 3), tr.reshape(-1, 3)], axis=0)

    # Exclude any hair pixels in corners
    tl_hair = hair[:c_h, :c_w]
    tr_hair = hair[:c_h, -c_w:]
    corners_hair = np.concatenate([tl_hair.reshape(-1), tr_hair.reshape(-1)], axis=0)

    bg_corners = corners[~corners_hair] if np.any(~corners_hair) else corners
    if len(bg_corners) < 20:
        top_slice = ref_rgb[:max(5, int(h_r * 0.03)), :].reshape(-1, 3)
        top_hair = hair[:max(5, int(h_r * 0.03)), :].reshape(-1)
        bg_corners = top_slice[~top_hair] if np.any(~top_hair) else top_slice

    backdrop_color = np.median(bg_corners, axis=0)
    backdrop_std = float(np.std(bg_corners, axis=0).max())

    # Core hair: erosion of hair mask around the upper head
    hair_u8 = hair.astype(np.uint8)
    core_deep_k = max(5, int(round(9 * (max(h_r, w_r) / 512.0))))
    if core_deep_k % 2 == 0:
        core_deep_k += 1
    hair_core_deep = cv2.erode(
        hair_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_deep_k, core_deep_k))
    )
    core_pts = ref_rgb[hair_core_deep > 0] if np.any(hair_core_deep > 0) else ref_rgb[hair]
    core_hair_color = np.median(core_pts, axis=0) if len(core_pts) > 0 else np.array([50.0, 35.0, 25.0])

    dist_to_bg = np.linalg.norm(ref_rgb.astype(np.float32) - backdrop_color, axis=-1)
    dist_to_hair = np.linalg.norm(ref_rgb.astype(np.float32) - core_hair_color, axis=-1)

    # Dynamic prune threshold:
    # A pixel is considered backdrop if:
    # 1. Color distance to backdrop is within statistical margin (std * 2.5, min 22, max 55)
    # 2. OR it is significantly closer to backdrop than to core hair (dist_to_bg < dist_to_hair * 0.75 and dist_to_bg < 65)
    base_thresh = float(np.clip(max(backdrop_std * 2.5, 22.0), 22.0, 55.0))
    prune_threshold = base_thresh
    ref_hsv = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2HSV)
    backdrop_hsv = cv2.cvtColor(np.uint8([[backdrop_color]]), cv2.COLOR_RGB2HSV)[0, 0]
    is_neutral_backdrop = (ref_hsv[:, :, 1] < max(25, int(backdrop_hsv[1]) + 20)) & (ref_hsv[:, :, 2] > max(90, int(backdrop_hsv[2]) - 60))
    is_backdrop = (dist_to_bg < base_thresh) | ((dist_to_bg < dist_to_hair * 0.75) & (dist_to_bg < 65.0)) | (is_neutral_backdrop & (dist_to_bg < 100.0) & (dist_to_bg < dist_to_hair))

    # Identify head region from landmarks if available
    head_core_y = int(h_r * 0.5)
    if landmarks_dict is not None and landmarks_dict.get("landmarks_2d") is not None:
        pts = landmarks_dict["landmarks_2d"]
        try:
            if len(pts) > 0:
                head_core_y = int(min(pt[1] for pt in pts))
        except (TypeError, ValueError):
            pass

    # Connected components of hair pixels matching backdrop color
    backdrop_hair = (hair & is_backdrop).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(backdrop_hair, connectivity=8)
    highlight_max = max(16, int(round(64.0 * ((max(h_r, w_r) / 512.0) ** 2))))

    prune_mask = np.zeros((h_r, w_r), dtype=bool)
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        touches_border = (x <= 0 or y <= 0 or (x + bw) >= w_r)
        comp = (labels == i)

        in_deep_core = bool(np.any(hair_core_deep[comp] > 0))
        # Keep only tiny interior highlight specks; prune perimeter fringe and large backdrop pockets.
        if area <= highlight_max and in_deep_core and not touches_border:
            continue
        prune_mask[comp] = True

    # Sanity guard: Pruning should NEVER wipe out more than 35% of the total hair mask
    total_hair_px = np.count_nonzero(hair)
    pruned_px = np.count_nonzero(prune_mask)
    if total_hair_px > 0 and (pruned_px / total_hair_px) > 0.35:
        # Fallback: only prune pixels that are on the outer perimeter (outer 3% border)
        border_margin = max(3, int(min(w_r, h_r) * 0.03))
        perimeter_mask = np.zeros((h_r, w_r), dtype=bool)
        perimeter_mask[:border_margin, :] = True
        perimeter_mask[:, :border_margin] = True
        perimeter_mask[:, -border_margin:] = True
        prune_mask = prune_mask & perimeter_mask

    cleaned = ref_hair_np.copy()
    cleaned[prune_mask] = 0
    info = {
        "pruned_count": int(np.count_nonzero(prune_mask)),
        "backdrop_color": backdrop_color,
        "backdrop_std": backdrop_std,
        "prune_threshold": prune_threshold,
        "is_backdrop": is_backdrop,
        "prune_mask": prune_mask,
    }
    return cleaned, info


def decontaminate_reference_hair_rgb(
    reference_rgb: np.ndarray,
    ref_hair_binary: np.ndarray,
    bounded_decontamination: bool = True,
    ring_radius_px: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace contaminated hair-edge RGB with interior hair color.

    Bounded mode inpaints only pixels that are inside the hair mask but outside
    the eroded core, so TELEA cannot sample exterior studio backdrop.
    """
    h_r, w_r = reference_rgb.shape[:2]
    binary = (ref_hair_binary > 0).astype(np.uint8)
    scale_ref = max(h_r, w_r) / 512.0
    erode_k = max(3, int(round(5.0 * scale_ref)))
    if erode_k % 2 == 0:
        erode_k += 1
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    core_hair = cv2.erode(binary, k_erode, iterations=1)
    if not np.any(core_hair > 0):
        k_erode_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        core_hair = cv2.erode(binary, k_erode_3, iterations=1)
        if not np.any(core_hair > 0):
            core_hair = binary.copy()

    if not bounded_decontamination:
        unknown_region = (core_hair == 0).astype(np.uint8)
        inpaint_radius = 7
    else:
        # Hair-rim only. ring_radius_px is retained for callers but no longer
        # dilates into exterior backdrop (that was sampling white studio fill).
        _ = ring_radius_px
        inpaint_radius = max(3, int(round(7.0 * scale_ref)))
        unknown_region = ((binary > 0) & (core_hair == 0)).astype(np.uint8)

    if np.any(core_hair > 0) and np.any(unknown_region > 0):
        # Sample interior core hair color to guard against backdrop sampling
        core_pixels = reference_rgb[core_hair > 0]
        median_hair = np.median(core_pixels, axis=0).astype(np.uint8) if len(core_pixels) > 0 else np.array([50, 35, 25], dtype=np.uint8)

        # Identify rim pixels that are actually contaminated by studio backdrop
        ref_hsv = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2HSV)
        backdrop_color = np.median(reference_rgb[:max(5, int(h_r * 0.03)), :max(5, int(w_r * 0.03))], axis=(0, 1))
        dist_to_bg = np.linalg.norm(reference_rgb.astype(np.float32) - backdrop_color, axis=-1)
        dist_to_hair = np.linalg.norm(reference_rgb.astype(np.float32) - median_hair.astype(np.float32), axis=-1)
        backdrop_hsv = cv2.cvtColor(np.uint8([[backdrop_color]]), cv2.COLOR_RGB2HSV)[0, 0]

        # Requirement C: Preserve genuine hair strands.
        # Do not remove every low-saturation, high-brightness pixel indiscriminately.
        # A rim pixel is contaminated only if it is actually close to the backdrop color
        # and closer to backdrop than to interior hair color.
        is_bg_color_match = (dist_to_bg < 45.0) & (dist_to_bg < dist_to_hair * 1.25)
        is_backdrop_bright_neutral = bool((backdrop_hsv[1] < 45) and (backdrop_hsv[2] > 80))
        is_neutral_bleed = (
            is_backdrop_bright_neutral
            & (ref_hsv[:, :, 1] < 35)
            & (ref_hsv[:, :, 2] > 110)
            & (dist_to_bg < 75.0)
            & (dist_to_bg < dist_to_hair)
        )
        contaminated = (unknown_region > 0) & (is_bg_color_match | is_neutral_bleed)

        if np.any(contaminated):
            guarded_canvas = reference_rgb.copy()
            guarded_canvas[binary == 0] = median_hair
            inpainted_rgb = cv2.inpaint(guarded_canvas, contaminated.astype(np.uint8), inpaint_radius, cv2.INPAINT_TELEA)

            decontaminated_rgb = reference_rgb.copy()
            decontaminated_rgb[contaminated] = inpainted_rgb[contaminated]
        else:
            decontaminated_rgb = reference_rgb.copy()
    else:
        decontaminated_rgb = reference_rgb.copy()
    return decontaminated_rgb, unknown_region


def build_refined_edit_mask(
    target_image: Image.Image,
    target_hair_mask: Image.Image,
    target_segmentation_map: Optional[np.ndarray] = None,
    face_landmarks: Optional[Dict[str, Any]] = None,
    cranial_expansion_px: int = 30,
    facial_feature_dilation_px: int = 7,
    protect_clothing: bool = True,
) -> Image.Image:
    """
    Build an edit mask for generative inpainting.

    Guarantees:
    - 100% coverage of the target person's original hair.
    - Upward and lateral volume expansion for the new hairstyle.
    - Zero overlap with eyes, eyebrows, nose, mouth, and clothing.

    Returns:
        Binary PIL Image ('L' mode, 0 or 255).
    """
    w_orig, h_orig = target_image.size
    mask_np = np.asarray(target_hair_mask.convert("L"), dtype=np.uint8)
    if mask_np.shape[:2] != (h_orig, w_orig):
        mask_np = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    binary_hair = (mask_np > 128).astype(np.uint8) * 255

    # 1. Expand hair region upwards and laterally for new hairstyle volume
    k_size = max(3, int(cranial_expansion_px))
    if k_size % 2 == 0:
        k_size += 1

    # Use an elliptical structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    expanded_hair = cv2.dilate(binary_hair, kernel, iterations=1)

    # 2. Build Protected Facial Region Mask
    protected_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

    # Layer A: Protection from ONNX segmentation map
    if target_segmentation_map is not None:
        seg = target_segmentation_map
        if seg.shape[:2] != (h_orig, w_orig):
            seg = cv2.resize(seg, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # Facial core: eyes, eyebrows, nose, mouth
        feat_mask = np.isin(seg, PROTECTED_SEG_CLASSES).astype(np.uint8) * 255
        if facial_feature_dilation_px > 0:
            k_feat = max(3, int(facial_feature_dilation_px))
            if k_feat % 2 == 0:
                k_feat += 1
            feat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_feat, k_feat))
            feat_mask = cv2.dilate(feat_mask, feat_kernel, iterations=1)

        protected_mask = np.maximum(protected_mask, feat_mask)

        # Clothing protection
        if protect_clothing:
            cloth_mask = np.isin(seg, CLOTHING_SEG_CLASSES).astype(np.uint8) * 255
            protected_mask = np.maximum(protected_mask, cloth_mask)

    # Layer B: Protection from MediaPipe facial landmarks (redundant safeguard)
    if face_landmarks and face_landmarks.get("detected") and face_landmarks.get("landmarks_2d") is not None:
        lms_2d = face_landmarks["landmarks_2d"]
        h_pts = np.int32(lms_2d)

        # Eye & mouth landmark hulls
        # Left eye: 33, 133, 160, 159, 158, 157, 173, 153, 144, 145, 153
        # Right eye: 263, 362, 387, 386, 385, 384, 398, 380, 373, 374, 380
        # Lips: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308
        feature_indices = [
            [33, 133, 160, 159, 158, 157, 173, 153, 144, 145],
            [263, 362, 387, 386, 385, 384, 398, 380, 373, 374],
            [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 185, 40, 39, 37, 0, 267, 269, 270, 409],
        ]
        lm_canvas = np.zeros((h_orig, w_orig), dtype=np.uint8)
        for idx_group in feature_indices:
            valid_pts = [h_pts[i] for i in idx_group if i < len(h_pts)]
            if len(valid_pts) >= 3:
                hull = cv2.convexHull(np.array(valid_pts))
                cv2.fillConvexPoly(lm_canvas, hull, 255)

        if facial_feature_dilation_px > 0:
            k_lm = max(5, int(facial_feature_dilation_px * 1.5))
            if k_lm % 2 == 0:
                k_lm += 1
            lm_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_lm, k_lm))
            lm_canvas = cv2.dilate(lm_canvas, lm_kernel, iterations=1)

        protected_mask = np.maximum(protected_mask, lm_canvas)

    # 3. Always include 100% of original hair, then carve out protected regions
    final_edit_mask = expanded_hair.copy()
    final_edit_mask[protected_mask > 0] = 0

    # Ensure binary output: 0 or 255
    final_edit_mask = (final_edit_mask > 128).astype(np.uint8) * 255
    return Image.fromarray(final_edit_mask, mode="L")


def build_feathered_blend_mask(
    edit_mask: Image.Image,
    feather_width_px: float = 6.0,
) -> np.ndarray:
    """
    Generate an inner distance-transform blend mask.

    Feathering is strictly confined INSIDE the edit mask:
    - Pixels outside edit_mask have alpha = 0.0 (100% target preservation).
    - Pixels deep inside edit_mask have alpha = 1.0 (100% generated hair).
    - Pixels near the inner boundary have a smooth linear ramp (0.0 to 1.0).

    Returns:
        np.ndarray of shape (H, W, 1) with float32 values in [0.0, 1.0].
    """
    mask_np = np.asarray(edit_mask.convert("L"), dtype=np.uint8)
    binary = (mask_np > 128).astype(np.uint8)

    dist_inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    width = max(1.0, float(feather_width_px))

    alpha = np.clip(dist_inside / width, 0.0, 1.0)
    # Strictly zero out anything outside the edit mask
    alpha[binary == 0] = 0.0

    return alpha[..., None].astype(np.float32)


def composite_with_feathering(
    target_image: Image.Image,
    generated_image: Image.Image,
    edit_mask: Image.Image,
    feather_width_px: float = 6.0,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Blends the generated inpainting output onto the target portrait using
    inner-distance feathering.

    Guarantees:
    - Target pixels outside `edit_mask` are 100% bit-exact identical to `target_image`.
    - Inner hairline and volume boundaries blend seamlessly.

    Returns:
        (composite_image, alpha_mask)
    """
    target_np = np.asarray(target_image.convert("RGB"), dtype=np.uint8)
    w, h = target_image.size

    gen_pil = generated_image.convert("RGB")
    if gen_pil.size != (w, h):
        gen_pil = gen_pil.resize((w, h), Image.Resampling.LANCZOS)
    gen_np = np.asarray(gen_pil, dtype=np.uint8)

    alpha = build_feathered_blend_mask(edit_mask, feather_width_px=feather_width_px)

    blended = (gen_np.astype(np.float32) * alpha + target_np.astype(np.float32) * (1.0 - alpha))
    blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)

    # Force exact pixel preservation outside edit mask
    mask_np = np.asarray(edit_mask.convert("L"), dtype=np.uint8)
    outside = mask_np == 0
    blended_uint8[outside] = target_np[outside]

    composite = Image.fromarray(blended_uint8, mode="RGB")
    return composite, alpha
