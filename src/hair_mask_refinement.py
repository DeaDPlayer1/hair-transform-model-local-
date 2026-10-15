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
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Prune studio-backdrop pixels that the hair parser labeled as hair.

    Keeps real interior highlights (small enclosed backdrop-like blobs) and
    dark hair. Removes large curl-gap pockets and perimeter fringe that match
    the studio backdrop. Does not globally strip bright pixels.
    """
    ref_rgb = reference_rgb
    ref_hair_np = reference_hair_mask
    h_r, w_r = ref_rgb.shape[:2]
    if ref_hair_np.shape[:2] != (h_r, w_r):
        ref_hair_np = cv2.resize(ref_hair_np, (w_r, h_r), interpolation=cv2.INTER_NEAREST)

    hair = ref_hair_np > 128
    border_px = max(10, int(round(15 * (max(h_r, w_r) / 512.0))))
    top_b = ref_rgb[:border_px, :]
    left_b = ref_rgb[:, :border_px]
    right_b = ref_rgb[:, -border_px:]
    perim_rgb = np.concatenate(
        [top_b.reshape(-1, 3), left_b.reshape(-1, 3), right_b.reshape(-1, 3)], axis=0
    )
    perim_hair = np.concatenate(
        [
            hair[:border_px].reshape(-1),
            hair[:, :border_px].reshape(-1),
            hair[:, -border_px:].reshape(-1),
        ]
    )
    bg_perim = perim_rgb[~perim_hair]
    if len(bg_perim) > 30:
        backdrop_color = np.median(bg_perim, axis=0)
        backdrop_std = float(np.std(bg_perim, axis=0).max())
    else:
        corners = np.concatenate([ref_rgb[:25, :25], ref_rgb[:25, -25:]], axis=1)
        backdrop_color = corners.mean(axis=(0, 1))
        backdrop_std = float(corners.std(axis=(0, 1)).max())

    color_dist = np.linalg.norm(ref_rgb.astype(np.float32) - backdrop_color, axis=-1)
    prune_threshold = max(24.0, backdrop_std * 3.5)
    is_backdrop = color_dist < prune_threshold

    hair_u8 = hair.astype(np.uint8)
    core_deep_k = max(5, int(round(7 * (max(h_r, w_r) / 512.0))))
    if core_deep_k % 2 == 0:
        core_deep_k += 1
    hair_core_deep = cv2.erode(
        hair_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_deep_k, core_deep_k))
    )

    backdrop_hair = (hair & is_backdrop).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(backdrop_hair, connectivity=8)
    highlight_max = max(16, int(round(48.0 * ((max(h_r, w_r) / 512.0) ** 2))))

    prune_mask = np.zeros((h_r, w_r), dtype=bool)
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        touches_border = x <= 0 or y <= 0 or (x + bw) >= w_r or (y + bh) >= h_r
        component = labels == i
        in_deep_core = bool(np.any(hair_core_deep[component] > 0))
        # Keep only tiny interior highlight specks; prune fringe and large pockets.
        if area <= highlight_max and in_deep_core and not touches_border:
            continue
        prune_mask[component] = True

    cleaned = ref_hair_np.copy()
    cleaned[prune_mask] = 0
    info = {
        "pruned_count": int(np.count_nonzero(prune_mask)),
        "backdrop_color": backdrop_color,
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
        decontaminated_rgb = cv2.inpaint(reference_rgb, unknown_region, inpaint_radius, cv2.INPAINT_TELEA)
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
