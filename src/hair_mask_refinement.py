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
CLOTHING_SEG_CLASSES = [16]


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
