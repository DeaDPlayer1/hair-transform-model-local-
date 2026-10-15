"""
Audited, landmark-guided 2D hairstyle transfer pipeline with:
- Consistent mirrored coordinate systems and bilateral symmetry mapping
- Anatomical head envelope constraints (no spill over shoulders or clothing)
- Protection of facial features and natural forehead hairline
- Selective inpainting without destructive bald/white holes
- Generation of 9 intermediate diagnostic images + hairline overlay
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from src.face_landmarks import (
    detect_face_landmarks,
    get_mirrored_landmarks,
    build_symmetric_correspondence,
    analyze_pose_and_mirroring,
    create_head_constraints,
    create_target_hairline_overlay,
    ANCHOR_LANDMARK_INDICES,
)


def _to_bgr_array(image):
    """Convert a PIL image to uint8 BGR numpy array."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def _to_binary_mask(mask, size):
    """Resize mask to (width, height) and threshold to binary (0 or 255)."""
    mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
    if (mask_array.shape[1], mask_array.shape[0]) != size:
        mask_array = cv2.resize(
            mask_array,
            size,
            interpolation=cv2.INTER_NEAREST,
        )
    return (mask_array >= 128).astype(np.uint8) * 255


def _mask_bbox(mask):
    """Return x, y, width, height for nonzero pixels."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2 - x1, y2 - y1


def _harmonize_hair_color(
    donor_bgr: np.ndarray,
    donor_hair_mask: np.ndarray,
    target_bgr: np.ndarray,
    target_hair_mask: np.ndarray,
    strength: float = 0.70,
) -> np.ndarray:
    """
    Match color and lighting temperature of donor hair to target in CIE LAB space.
    Preserves donor hair details while harmonizing lightness and tone.
    """
    donor_indices = donor_hair_mask > 0
    if not np.any(donor_indices):
        return donor_bgr

    donor_lab = cv2.cvtColor(donor_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    tgt_indices = target_hair_mask > 0
    if np.count_nonzero(tgt_indices) > 50:
        tgt_sample = target_lab[tgt_indices]
    else:
        h = target_bgr.shape[0]
        tgt_sample = target_lab[: int(h * 0.5), :]

    dnr_sample = donor_lab[donor_indices]

    result_lab = donor_lab.copy()
    for ch in range(3):
        m_tgt = float(tgt_sample[:, ch].mean())
        s_tgt = float(max(1e-3, tgt_sample[:, ch].std()))

        m_dnr = float(dnr_sample[:, ch].mean())
        s_dnr = float(max(1e-3, dnr_sample[:, ch].std()))

        ratio = np.clip(s_tgt / s_dnr, 0.65, 1.35)
        matched = (donor_lab[:, :, ch] - m_dnr) * ratio + m_tgt
        result_lab[:, :, ch] = np.where(
            donor_indices,
            (1.0 - strength) * donor_lab[:, :, ch] + strength * matched,
            donor_lab[:, :, ch],
        )

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


def transfer_hairstyle(
    target_image,
    reference_image,
    target_hair_mask,
    reference_hair_mask,
    mirror_reference=None,
    match_color: bool = True,
    hairline_offset_y: int = 0,
    scale_multiplier: float = 1.0,
    method: str = "landmark_affine",
    return_diagnostics: bool = False,
    target_segmentation_map: np.ndarray = None,
    save_intermediate_dir: Path = None,
):
    """
    Audited landmark-guided hairstyle transfer with head envelope constraints.

    Returns:
        If return_diagnostics is False:
            PIL Image of composite result.
        If return_diagnostics is True:
            (PIL Image, dict of diagnostics including intermediate images).
    """
    target_bgr = _to_bgr_array(target_image)
    ref_bgr = _to_bgr_array(reference_image)

    target_h, target_w = target_bgr.shape[:2]
    ref_h, ref_w = ref_bgr.shape[:2]

    # Clean masks
    tgt_mask = _to_binary_mask(target_hair_mask, (target_w, target_h))
    ref_mask = _to_binary_mask(reference_hair_mask, (ref_w, ref_h))

    # 1. Face landmark detection and pose analysis
    tgt_lms_info = detect_face_landmarks(target_bgr)
    ref_lms_info = detect_face_landmarks(ref_bgr)

    pose_analysis = analyze_pose_and_mirroring(tgt_lms_info, ref_lms_info)

    # Determine mirroring
    if mirror_reference is None:
        should_mirror = pose_analysis["recommend_mirror"]
    else:
        should_mirror = bool(mirror_reference)

    # 2. Consistent coordinate mirroring
    # If mirrored, mirror image, mask, and landmark coordinates consistently
    if should_mirror:
        active_ref_bgr = cv2.flip(ref_bgr, 1)
        active_ref_mask = cv2.flip(ref_mask, 1)
        active_ref_lms = get_mirrored_landmarks(ref_lms_info.get("landmarks_2d"), ref_w)
        active_ref_yaw = -pose_analysis["reference_yaw"]
    else:
        active_ref_bgr = ref_bgr.copy()
        active_ref_mask = ref_mask.copy()
        active_ref_lms = ref_lms_info.get("landmarks_2d")
        active_ref_yaw = pose_analysis["reference_yaw"]

    active_yaw_diff = abs(pose_analysis["target_yaw"] - active_ref_yaw)

    # Diagnostic metadata dictionary
    diagnostics = {
        "method": method,
        "target_yaw": pose_analysis["target_yaw"],
        "reference_yaw": active_ref_yaw,
        "is_mirrored": should_mirror,
        "yaw_mismatch": active_yaw_diff,
        "pose_warning": pose_analysis["pose_mismatch_warning"],
        "scale": 1.0,
        "rotation_deg": 0.0,
        "tx": 0.0,
        "ty": 0.0,
        "inliers": 0,
        "alignment_mode": "fallback",
        "color_matched": match_color,
        "spill_pixels": 0,
        "spill_percentage": 0.0,
        "clothing_overlap_px": 0,
        "uncovered_hair_px": 0,
        "uncovered_hair_percentage": 0.0,
        "warnings": [],
        "intermediates": {},
    }

    if pose_analysis["pose_mismatch_warning"]:
        diagnostics["warnings"].append(pose_analysis["pose_mismatch_warning"])

    # 3. Target head envelope and protected face constraints
    permitted_envelope, protected_face_mask = create_head_constraints(
        (target_h, target_w),
        tgt_lms_info,
        segmentation_map=target_segmentation_map,
        target_hair_mask=tgt_mask,
    )

    # 4. Baseline mode fallback if requested
    if method == "baseline_bbox":
        tx, ty, tw, th = _mask_bbox(tgt_mask)
        rx, ry, rw, rh = _mask_bbox(active_ref_mask)

        if tw == 0 or th == 0 or rw == 0 or rh == 0:
            raise ValueError("Empty hair mask detected.")

        donor_crop = active_ref_bgr[ry:ry + rh, rx:rx + rw]
        donor_mask_crop = active_ref_mask[ry:ry + rh, rx:rx + rw]

        donor_resized = cv2.resize(donor_crop, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        donor_mask_resized = cv2.resize(donor_mask_crop, (tw, th), interpolation=cv2.INTER_NEAREST)

        removal_mask = cv2.dilate(tgt_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        base_cleaned = cv2.inpaint(target_bgr, removal_mask, 5, cv2.INPAINT_TELEA)

        x1 = max(0, tx)
        y1 = max(0, ty)
        x2 = min(target_w, tx + tw)
        y2 = min(target_h, ty + th)

        donor_x1 = x1 - tx
        donor_y1 = y1 - ty
        donor_x2 = donor_x1 + (x2 - x1)
        donor_y2 = donor_y1 + (y2 - y1)

        d_reg = donor_resized[donor_y1:donor_y2, donor_x1:donor_x2]
        d_mask_reg = donor_mask_resized[donor_y1:donor_y2, donor_x1:donor_x2]

        blur_size = max(3, int(round(min(target_h, target_w) * 0.025)))
        if blur_size % 2 == 0:
            blur_size += 1
        alpha = cv2.GaussianBlur(d_mask_reg.astype(np.float32) / 255.0, (blur_size, blur_size), 0)[..., None]

        res = base_cleaned.copy()
        t_reg = res[y1:y2, x1:x2].astype(np.float32)
        blended = d_reg.astype(np.float32) * alpha + t_reg * (1.0 - alpha)
        res[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        diagnostics["alignment_mode"] = "baseline_bbox"
        res_pil = Image.fromarray(res_rgb, mode="RGB")
        return (res_pil, diagnostics) if return_diagnostics else res_pil

    # 5. Affine Transform Estimation
    M = None
    inliers = 0

    if tgt_lms_info["detected"] and (active_ref_lms is not None):
        tgt_lms = tgt_lms_info["landmarks_2d"]

        # Build correspondence with bilateral symmetry swapping
        src_pts, dst_pts = build_symmetric_correspondence(
            active_ref_lms,
            tgt_lms,
            is_src_mirrored=should_mirror,
            anchor_indices=ANCHOR_LANDMARK_INDICES,
        )

        M, inlier_mask = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=15.0,
            maxIters=2000,
        )

        if M is not None:
            inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else len(src_pts)
            scale = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
            rot_deg = float(np.arctan2(M[1, 0], M[0, 0]) * 180.0 / np.pi)

            # Reject implausible transformations (e.g. scale too extreme or too few inliers)
            if inliers >= 5 and 0.05 <= scale <= 10.0 and abs(rot_deg) <= 45.0:
                diagnostics["scale"] = scale
                diagnostics["rotation_deg"] = rot_deg
                diagnostics["tx"] = float(M[0, 2])
                diagnostics["ty"] = float(M[1, 2])
                diagnostics["inliers"] = inliers
                diagnostics["alignment_mode"] = "landmark_affine"
            else:
                diagnostics["warnings"].append(
                    f"Affine estimation rejected (scale={scale:.2f}, rot={rot_deg:.1f}°, inliers={inliers}). "
                    "Falling back to cranial bounding geometry."
                )
                M = None

    # Fallback to mask cranial geometry if landmarks failed or were rejected
    if M is None:
        diagnostics["alignment_mode"] = "mask_geometry_fallback"
        rx, ry, rw, rh = _mask_bbox(active_ref_mask)
        tx, ty, tw, th = _mask_bbox(tgt_mask)
        tw_eff = tw if tw > 0 else int(target_w * 0.5)
        scale = (tw_eff / float(max(1, rw))) * scale_multiplier
        dx = (tx if tw > 0 else int(target_w * 0.25)) - (rx * scale)
        dy = (ty if th > 0 else int(target_h * 0.15)) - (ry * scale) + hairline_offset_y
        M = np.array([[scale, 0.0, dx], [0.0, scale, dy]], dtype=np.float32)
        diagnostics["scale"] = scale
        diagnostics["tx"] = dx
        diagnostics["ty"] = dy

    # Apply manual scale fine-tuning around target center if requested
    if scale_multiplier != 1.0:
        cx = target_w / 2.0
        cy = target_h / 2.0
        if tgt_lms_info.get("detected"):
            cx = float(tgt_lms_info["landmarks_2d"][168, 0])
            cy = float(tgt_lms_info["landmarks_2d"][168, 1])
        S = np.array([
            [scale_multiplier, 0, (1 - scale_multiplier) * cx],
            [0, scale_multiplier, (1 - scale_multiplier) * cy],
            [0, 0, 1]
        ], dtype=np.float64)
        M_homo = np.vstack([M, [0, 0, 1]])
        M = (S @ M_homo)[:2, :]

    if hairline_offset_y != 0:
        M[1, 2] += hairline_offset_y

    # 6. Warp donor image and donor hair mask
    warped_donor_bgr = cv2.warpAffine(
        active_ref_bgr,
        M,
        (target_w, target_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    warped_donor_mask = cv2.warpAffine(
        active_ref_mask,
        M,
        (target_w, target_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # 7. Numerical Validation: Spill and Overlap metrics
    total_warped_px = int(np.count_nonzero(warped_donor_mask))
    spill_px = int(np.count_nonzero(warped_donor_mask & (permitted_envelope == 0)))
    clothing_overlap = 0
    if target_segmentation_map is not None:
        clothing_overlap = int(np.count_nonzero(warped_donor_mask & (target_segmentation_map == 16)))

    spill_pct = (spill_px / max(1, total_warped_px)) * 100.0
    diagnostics["spill_pixels"] = spill_px
    diagnostics["spill_percentage"] = spill_pct
    diagnostics["clothing_overlap_px"] = clothing_overlap

    if spill_pct > 25.0:
        diagnostics["warnings"].append(
            f"{spill_pct:.1f}% of donor hair extended outside the plausible head envelope (clipped to prevent shoulder/body spill)."
        )

    # 8. Clip donor hair to permitted head envelope
    effective_donor_mask = (warped_donor_mask > 128).astype(np.uint8) * 255
    effective_donor_mask[permitted_envelope == 0] = 0

    # 9. Inpainting & Hole Prevention
    # Check uncovered target hair
    uncovered_old_hair = (tgt_mask > 0) & (effective_donor_mask == 0)
    uncovered_px = int(np.count_nonzero(uncovered_old_hair))
    tgt_hair_total = int(np.count_nonzero(tgt_mask))
    uncovered_pct = (uncovered_px / max(1, tgt_hair_total)) * 100.0
    diagnostics["uncovered_hair_px"] = uncovered_px
    diagnostics["uncovered_hair_percentage"] = uncovered_pct

    if uncovered_pct > 20.0:
        diagnostics["warnings"].append(
            f"Partial head coverage: {uncovered_pct:.1f}% of target hair region is not covered by donor. "
            "Existing hair is preserved in uncovered areas to prevent scalp/background holes."
        )

    # Inpaint ONLY true background flyaways outside the target head boundary.
    # NEVER inpaint inside the target head, forehead, or uncovered scalp!
    # A flyaway pixel is target hair that is:
    # 1. Uncovered by new hair
    # 2. Outside the main cranial envelope (protruding into background)
    # 3. Not touching the face
    flyaway_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    if tgt_hair_total > 0:
        # Detect pixels where target hair protrudes outside permitted envelope into background
        flyaways = (tgt_mask > 0) & (permitted_envelope == 0) & (effective_donor_mask == 0)
        if target_segmentation_map is not None:
            # Don't inpaint skin, cloth, or features
            flyaways = flyaways & (target_segmentation_map != 1) & (target_segmentation_map != 16)
        if np.any(flyaways):
            k_inpaint = max(3, int(round(target_h * 0.015)))
            if k_inpaint % 2 == 0:
                k_inpaint += 1
            flyaway_mask = cv2.dilate(
                flyaways.astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_inpaint, k_inpaint)),
                iterations=1,
            )

    if np.any(flyaway_mask > 0):
        inpaint_radius = max(3, int(round(target_h * 0.01)))
        base_image = cv2.inpaint(target_bgr, flyaway_mask, inpaint_radius, cv2.INPAINT_TELEA)
    else:
        base_image = target_bgr.copy()

    # 10. Color & Lighting Harmonization (LAB Space)
    if match_color:
        harmonized_donor = _harmonize_hair_color(
            warped_donor_bgr,
            effective_donor_mask,
            target_bgr,
            tgt_mask,
            strength=0.70,
        )
    else:
        harmonized_donor = warped_donor_bgr

    # 11. Distance-Transform Edge Feathering
    dist = cv2.distanceTransform(
        (effective_donor_mask > 0).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    feather_px = max(2.5, min(8.0, target_h * 0.012))
    alpha = np.clip(dist / feather_px, 0.0, 1.0)
    # Soft gaussian smoothing along the transition
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)[..., None]

    # 12. Alpha Blending: Donor hair onto base image
    composite = (
        harmonized_donor.astype(np.float32) * alpha
        + base_image.astype(np.float32) * (1.0 - alpha)
    )
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    # 13. Build 9 Intermediate Diagnostic Images + Hairline Overlay
    hairline_overlay = create_target_hairline_overlay(
        target_bgr,
        tgt_lms_info,
        segmentation_map=target_segmentation_map,
        permitted_envelope=permitted_envelope,
    )

    intermediates = {
        "1_target": target_bgr,
        "2_donor": ref_bgr,
        "3_target_hair_mask": tgt_mask,
        "4_donor_hair_mask": active_ref_mask,
        "5_transformed_donor_image": warped_donor_bgr,
        "6_transformed_donor_mask": effective_donor_mask,
        "7_protected_face_mask": protected_face_mask,
        "8_inpainting_result": base_image,
        "9_final_composite": composite,
        "hairline_overlay": hairline_overlay,
    }
    diagnostics["intermediates"] = intermediates

    # Save to disk if directory specified
    if save_intermediate_dir is not None:
        save_dir = Path(save_intermediate_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, img_arr in intermediates.items():
            out_file = save_dir / f"diag_{name}.png"
            cv2.imwrite(str(out_file), img_arr)

    composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(composite_rgb, mode="RGB")

    return (result_pil, diagnostics) if return_diagnostics else result_pil


def save_transfer_result(image, output_path):
    """Save a transfer result, creating parent directories if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
