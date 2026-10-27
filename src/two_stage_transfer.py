"""
Two-Stage Hairstyle Transfer Orchestration Module for HairTryOn.

Stage 1: Converts target portrait to a natural bald/scalp base image.
Stage 2: Aligns and transfers reference hairstyle onto the bald base using
         IP-Adapter reference conditioning and strict identity preservation.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.bald_converter import (
    BaldConverterConfig,
    create_bald_converter,
    compute_inpaint_dimensions,
    compute_face_geometry,
)
from src.hair_mask_refinement import refine_reference_hair_mask
from src.hairstyle_alignment import (
    align_reference_to_target,
    EAR_SEG_CLASSES,
)
from src.local_generative_transfer import (
    load_local_generative_pipeline,
    prepare_reference_crop,
    get_process_memory_mb,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/two_stage_transfer.json")


def load_two_stage_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to parse config at {path}: {e}. Using defaults.")

    return {
        "stage1": {
            "backend": "auto",
            "num_inference_steps": 12,
            "guidance_scale": 7.5,
            "hair_mask_expansion_ratio": 0.035,
            "max_hair_mask_expansion_px": 32,
            "feather_width_px": 3.0,
        },
        "stage2": {
            "num_inference_steps": 8,
            "guidance_scale": 7.5,
            "ip_adapter_scale": 0.85,
            "cranial_expansion_px": 30,
            "feather_width_px": 6.0,
            "hairline_transition_px": 15,
            "donor_coverage_threshold": 0.75,
            "seed": 42,
        },
        "controlnet": {"enabled": "auto", "model_path": None},
        "safety": {
            "preserve_original_outside_hair_envelope": True,
            "preserve_identity_pixels": True,
            "fail_on_invalid_model_output": True,
        },
    }


def compute_preservation_metrics(
    original_target_np: np.ndarray,
    final_composite_np: np.ndarray,
    broad_envelope_np: np.ndarray,
    identity_protection_np: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute preservation metrics between original target and final composite."""
    outside_mask = (broad_envelope_np == 0)
    total_outside = int(np.count_nonzero(outside_mask))

    if total_outside > 0:
        tgt_outside = original_target_np[outside_mask].astype(np.float32)
        comp_outside = final_composite_np[outside_mask].astype(np.float32)
        diff_outside = np.abs(tgt_outside - comp_outside)
        max_diff = float(np.max(diff_outside))
        mean_diff = float(np.mean(diff_outside))
        exact_match = np.all(diff_outside < 1.0, axis=-1)
        preservation_pct = (float(np.count_nonzero(exact_match)) / total_outside) * 100.0
    else:
        max_diff = 0.0
        mean_diff = 0.0
        preservation_pct = 100.0

    leak_count = 0
    if identity_protection_np is not None:
        ident_mask = identity_protection_np > 0
        if np.any(ident_mask):
            diff_ident = np.abs(
                original_target_np[ident_mask].astype(np.float32)
                - final_composite_np[ident_mask].astype(np.float32)
            )
            leak_count = int(np.count_nonzero(np.any(diff_ident >= 1.0, axis=-1)))

    return {
        "outside_pixel_preservation_pct": round(preservation_pct, 4),
        "max_outside_pixel_diff": round(max_diff, 2),
        "mean_outside_pixel_diff": round(mean_diff, 4),
        "identity_protection_leak_count": leak_count,
    }


def pad_canvas_for_headroom(image: Image.Image, pad_top_px: int) -> Image.Image:
    """Add blank rows above the image so a voluminous warped hairstyle has
    somewhere to go instead of being hard-clipped at the canvas edge."""
    if pad_top_px <= 0:
        return image
    w, h = image.size
    arr = np.asarray(image.convert("RGB"))
    c = max(8, int(min(w, h) * 0.03))
    corner = np.concatenate([arr[:c, :c].reshape(-1, 3), arr[:c, -c:].reshape(-1, 3)])
    fill_color = tuple(int(x) for x in np.median(corner, axis=0))
    if image.mode == "RGBA":
        padded = Image.new("RGBA", (w, h + pad_top_px), fill_color + (255,))
        padded.paste(image, (0, pad_top_px))
    else:
        padded = Image.new("RGB", (w, h + pad_top_px), fill_color)
        padded.paste(image.convert("RGB"), (0, pad_top_px))
    return padded


def shift_landmarks(landmarks: Optional[Dict[str, Any]], dy: int) -> Optional[Dict[str, Any]]:
    if not landmarks or not landmarks.get("detected") or dy == 0:
        return landmarks
    shifted = dict(landmarks)
    lm2d = landmarks.get("landmarks_2d")
    if lm2d is not None:
        if isinstance(lm2d, np.ndarray):
            lm2d = lm2d.copy()
            lm2d[:, 1] += dy
            shifted["landmarks_2d"] = lm2d
        elif isinstance(lm2d, list):
            shifted["landmarks_2d"] = [(pt[0], pt[1] + dy) for pt in lm2d]

    lm3d = landmarks.get("landmarks_3d")
    if lm3d is not None:
        if isinstance(lm3d, np.ndarray):
            lm3d = lm3d.copy()
            lm3d[:, 1] += dy
            shifted["landmarks_3d"] = lm3d
        elif isinstance(lm3d, list):
            shifted["landmarks_3d"] = [
                (pt[0], pt[1] + dy, pt[2]) if len(pt) >= 3 else (pt[0], pt[1] + dy)
                for pt in lm3d
            ]
    return shifted


def describe_hair_color(rgb_array: np.ndarray) -> str:
    """Classifies primary hair color shade from core hair RGB pixels."""
    if rgb_array.size == 0:
        return ""
    px = rgb_array.reshape(-1, 3).astype(np.float32)
    lightness = float(px.mean())
    hsv = cv2.cvtColor(px.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
    hue = float(np.median(hsv[:, 0]))
    if lightness < 45:
        return "black"
    if lightness > 180 and hue < 35:
        return "blonde"
    if 5 <= hue <= 20 and lightness < 120:
        return "auburn reddish-brown"
    if lightness > 140:
        return "light brown"
    return "medium-to-dark brown"


def generate_two_stage_hairstyle_transfer(
    target_image: Image.Image,
    reference_image: Image.Image,
    pipeline=None,
    target_hair_mask: Optional[Image.Image] = None,
    reference_hair_mask: Optional[Image.Image] = None,
    target_segmentation_map: Optional[np.ndarray] = None,
    reference_segmentation_map: Optional[np.ndarray] = None,
    target_hair_prob: Optional[np.ndarray] = None,
    face_landmarks: Optional[Dict[str, Any]] = None,
    reference_landmarks: Optional[Dict[str, Any]] = None,
    config: Optional[Union[Dict[str, Any], Path, str]] = None,
    stage1_backend: str = "auto",
    stage1_steps: Optional[int] = None,
    stage2_steps: Optional[int] = None,
    ip_adapter_scale: Optional[float] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
    cranial_expansion_px: Optional[int] = None,
    feather_width_px: Optional[float] = None,
    hairline_transition_px: Optional[int] = None,
    donor_coverage_threshold: Optional[float] = None,
    use_controlnet: str = "auto",
    stage1_only: bool = False,
    bald_base: Optional[Image.Image] = None,
    headwear_mask: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
    detect_headwear: bool = True,
    pad_headroom: bool = True,
    save_experiment: bool = True,
    experiment_id: Optional[str] = None,
    output_dir: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """
    Run complete two-stage hairstyle transfer.

    Stage 1: Bald converter creates bald/scalp base portrait.
    Stage 2: Reference hairstyle aligned and inpainted onto bald base portrait.
    """
    t_start = time.perf_counter()
    warnings: List[str] = []

    # 1. Resolve configuration
    base_cfg = load_two_stage_config(config if isinstance(config, (str, Path)) else None)
    if isinstance(config, dict):
        base_cfg.update(config)

    s1_cfg = base_cfg.get("stage1", {})
    s2_cfg = base_cfg.get("stage2", {})

    pad_headroom_effective = pad_headroom and s2_cfg.get("pad_headroom", True)

    final_s1_backend = stage1_backend if stage1_backend != "auto" else s1_cfg.get("backend", "auto")
    final_s1_steps = stage1_steps if stage1_steps is not None else s1_cfg.get("num_inference_steps", 12)
    final_s2_steps = stage2_steps if stage2_steps is not None else s2_cfg.get("num_inference_steps", 8)
    final_scale = ip_adapter_scale if ip_adapter_scale is not None else s2_cfg.get("ip_adapter_scale", 0.85)
    final_guidance = guidance_scale if guidance_scale is not None else s2_cfg.get("guidance_scale", 7.5)
    final_seed = seed if seed is not None else s2_cfg.get("seed", 42)
    final_cranial = cranial_expansion_px if cranial_expansion_px is not None else s2_cfg.get("cranial_expansion_px", 30)
    final_feather = feather_width_px if feather_width_px is not None else s2_cfg.get("feather_width_px", 6.0)
    final_transition = hairline_transition_px if hairline_transition_px is not None else s2_cfg.get("hairline_transition_px", 15)
    final_threshold = donor_coverage_threshold if donor_coverage_threshold is not None else s2_cfg.get("donor_coverage_threshold", 0.75)

    # 2. Preprocess & Validate Target & Reference
    target_img_rgb = target_image.convert("RGB")
    ref_img_rgb = reference_image.convert("RGB") if reference_image is not None else target_img_rgb.copy()
    w_t, h_t = target_img_rgb.size
    w_r, h_r = ref_img_rgb.size

    # Segmentation
    if target_hair_mask is None or target_segmentation_map is None:
        from src.hair_segmentation import segment_hair as app_segment_hair

        t_seg_res = app_segment_hair(target_img_rgb, return_seg_map=True, return_hair_prob=True)
        if target_hair_mask is None:
            target_hair_mask = t_seg_res[0]
        if target_segmentation_map is None:
            target_segmentation_map = t_seg_res[4]
        if target_hair_prob is None:
            target_hair_prob = t_seg_res[5]

    if not stage1_only and reference_image is not None:
        from src.hair_segmentation import segment_hair as app_segment_hair

        if reference_hair_mask is None or reference_segmentation_map is None:
            r_seg_res = app_segment_hair(ref_img_rgb, return_seg_map=True)
            if reference_hair_mask is None:
                reference_hair_mask = r_seg_res[0]
            if reference_segmentation_map is None and len(r_seg_res) > 4:
                reference_segmentation_map = r_seg_res[4]
    elif reference_hair_mask is None:
        reference_hair_mask = Image.new("L", (w_r, h_r), 0)

    # Facial Landmarks
    if face_landmarks is None:
        from src.face_landmarks import detect_face_landmarks

        tgt_bgr = cv2.cvtColor(np.asarray(target_img_rgb), cv2.COLOR_RGB2BGR)
        face_landmarks = detect_face_landmarks(tgt_bgr)

    if not stage1_only and reference_landmarks is None and reference_image is not None:
        from src.face_landmarks import detect_face_landmarks

        ref_bgr = cv2.cvtColor(np.asarray(ref_img_rgb), cv2.COLOR_RGB2BGR)
        reference_landmarks = detect_face_landmarks(ref_bgr)

    # Ensure mask dimensions match
    target_hair_mask_clean = target_hair_mask.convert("L")
    if target_hair_mask_clean.size != (w_t, h_t):
        target_hair_mask_clean = target_hair_mask_clean.resize((w_t, h_t), Image.Resampling.NEAREST)

    ref_hair_mask_clean = reference_hair_mask.convert("L")
    if ref_hair_mask_clean.size != (w_r, h_r):
        ref_hair_mask_clean = ref_hair_mask_clean.resize((w_r, h_r), Image.Resampling.NEAREST)

    # Headroom canvas padding for voluminous styles (prevents flat top cropping)
    orig_w_t, orig_h_t = w_t, h_t
    pad_top_px = 0
    if pad_headroom_effective and min(w_t, h_t) >= 256:
        geom_for_pad = compute_face_geometry(face_landmarks, w_t, h_t)
        pad_top_px = int(np.clip(geom_for_pad["face_scale"] * 0.35, 40, 260))

        target_img_rgb = pad_canvas_for_headroom(target_img_rgb, pad_top_px)
        face_landmarks = shift_landmarks(face_landmarks, pad_top_px)
        if target_segmentation_map is not None:
            target_segmentation_map = cv2.copyMakeBorder(
                target_segmentation_map, pad_top_px, 0, 0, 0, cv2.BORDER_CONSTANT, value=0
            )
        if target_hair_prob is not None:
            target_hair_prob = cv2.copyMakeBorder(
                target_hair_prob, pad_top_px, 0, 0, 0, cv2.BORDER_CONSTANT, value=0.0
            )
        target_hair_mask_clean = ImageOps.expand(target_hair_mask_clean, border=(0, pad_top_px, 0, 0), fill=0)
        if bald_base is not None and bald_base.size == (orig_w_t, orig_h_t):
            bald_base = pad_canvas_for_headroom(bald_base, pad_top_px)
        w_t, h_t = target_img_rgb.size

    # 3. Model Pipeline Management (load once, reuse for both stages)
    shared_pipeline = pipeline
    if shared_pipeline is None:
        logger.info("TwoStageTransfer: Loading local generative pipeline (SD 1.5 + IP-Adapter Plus)...")
        shared_pipeline = load_local_generative_pipeline()

    # 4. ControlNet Inspection
    controlnet_active = False
    controlnet_reason = "No compatible pretrained ControlNet model weights found. Using landmark alignment + IP-Adapter reference guidance."
    if use_controlnet != "off":
        logger.info(f"ControlNet status: {controlnet_reason}")
    warnings.append(f"ControlNet: {controlnet_reason}")

    # =========================================================
    # STAGE 1: BALD BASE CONVERSION
    # =========================================================
    if bald_base is None:
        logger.info("TwoStageTransfer: Starting Stage 1 (Bald/Scalp Reconstruction)...")
        s1_t0 = time.perf_counter()

        s1_config = BaldConverterConfig(
            backend=final_s1_backend,
            num_inference_steps=final_s1_steps,
            guidance_scale=final_guidance,
            hair_mask_expansion_ratio=s1_cfg.get("hair_mask_expansion_ratio", BaldConverterConfig.hair_mask_expansion_ratio),
            max_hair_mask_expansion_px=s1_cfg.get("max_hair_mask_expansion_px", BaldConverterConfig.max_hair_mask_expansion_px),
            feather_width_px=s1_cfg.get("feather_width_px", BaldConverterConfig.feather_width_px),
            prompt=s1_cfg.get("prompt", BaldConverterConfig.prompt),
            negative_prompt=s1_cfg.get("negative_prompt", BaldConverterConfig.negative_prompt),
            onnx_model_path=s1_cfg.get("onnx_model_path", None),
            headwear_mask_path=headwear_mask if isinstance(headwear_mask, (str, Path)) else s1_cfg.get("headwear_mask_path", None),
            detect_headwear=detect_headwear,
            seed=final_seed,
        )

        bald_converter = create_bald_converter(s1_config, pipeline=shared_pipeline)
        s1_result = bald_converter.convert(
            target_image=target_img_rgb,
            target_hair_mask=target_hair_mask_clean,
            target_segmentation_map=target_segmentation_map,
            face_landmarks=face_landmarks,
            headwear_mask=headwear_mask,
            seed=final_seed,
            hair_prob=target_hair_prob,
        )
        s1_duration = time.perf_counter() - s1_t0

        bald_base = s1_result.bald_base
        stage1_edit_mask = s1_result.edit_mask
        stage1_protection_mask = s1_result.protection_mask
        stage1_removal_overlay = s1_result.removal_overlay
        stage1_headwear_mask = s1_result.headwear_mask
        stage1_raw_generated = s1_result.raw_generated
        stage1_bg_repair_mask = getattr(s1_result, "background_repair_mask", None)
        stage1_bg_repair_preview = getattr(s1_result, "background_repair_preview", None)
        stage1_bald_base_before_after = getattr(s1_result, "bald_base_before_after", None)
        stage1_bg_difference = getattr(s1_result, "background_difference", None)
        if s1_result.warnings:
            warnings.extend(s1_result.warnings)
    else:
        logger.info("TwoStageTransfer: Using provided pre-computed bald base.")
        s1_duration = 0.0
        stage1_edit_mask = Image.new("L", (w_t, h_t), 0)
        stage1_protection_mask = Image.new("L", (w_t, h_t), 0)
        stage1_removal_overlay = None
        stage1_headwear_mask = None
        stage1_raw_generated = None
        stage1_bg_repair_mask = None
        stage1_bg_repair_preview = None
        stage1_bald_base_before_after = None
        stage1_bg_difference = None

    # Validate Stage 1 outputs
    if bald_base.size != (w_t, h_t):
        raise ValueError(
            f"Stage 1 returned bald base with mismatched size {bald_base.size}, expected {(w_t, h_t)}"
        )

    # If stage1_only requested, save Stage 1 artifacts and exit before Stage 2
    if stage1_only:
        logger.info("TwoStageTransfer: Stage 1 complete. Early exit requested via stage1_only=True.")
        run_tag = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir:
            exp_dir = Path(output_dir) / f"{run_tag}"
        else:
            exp_dir = Path("outputs/two_stage_runs") / f"{run_tag}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_dir_path = str(exp_dir.resolve())

        target_img_rgb.save(exp_dir / "target_original.png")
        target_img_rgb.save(exp_dir / "input_target.png")
        target_hair_mask_clean.save(exp_dir / "target_hair_mask.png")
        stage1_edit_mask.save(exp_dir / "stage1_edit_mask.png")
        stage1_protection_mask.save(exp_dir / "strict_identity_protection.png")
        stage1_protection_mask.save(exp_dir / "stage1_protection_mask.png")
        if stage1_removal_overlay:
            stage1_removal_overlay.save(exp_dir / "stage1_removal_overlay.png")
        if stage1_headwear_mask:
            stage1_headwear_mask.save(exp_dir / "headwear_mask.png")
            stage1_headwear_mask.save(exp_dir / "stage1_headwear_mask.png")
        if stage1_raw_generated:
            stage1_raw_generated.save(exp_dir / "stage1_raw_generated.png")
        bald_base.save(exp_dir / "bald_base.png")
        if stage1_bg_repair_mask:
            stage1_bg_repair_mask.save(exp_dir / "stage1_background_repair_mask.png")
        if stage1_bg_repair_preview:
            stage1_bg_repair_preview.save(exp_dir / "stage1_background_repair_preview.png")
        if stage1_bald_base_before_after:
            stage1_bald_base_before_after.save(exp_dir / "stage1_bald_base_before_after.png")
        if stage1_bg_difference:
            stage1_bg_difference.save(exp_dir / "stage1_background_difference.png")

        s1_metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stage": "stage1_only",
            "target_dimensions": [w_t, h_t],
            "stage1_backend": final_s1_backend,
            "stage1_steps": final_s1_steps,
            "stage1_seed": final_seed,
            "headwear_detected": s1_result.headwear_detected,
            "duration_seconds": s1_duration,
            "stage1_details": s1_result.metadata,
        }
        with (exp_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(s1_metadata, f, indent=2)
        with (exp_dir / "stage1_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(s1_result.metadata, f, indent=2)
        with (exp_dir / "warnings.json").open("w", encoding="utf-8") as f:
            json.dump(warnings, f, indent=2)

        return {
            "bald_base": bald_base,
            "stage1_raw_generated": stage1_raw_generated,
            "stage1_edit_mask": stage1_edit_mask,
            "stage1_removal_overlay": stage1_removal_overlay,
            "stage1_protection_mask": stage1_protection_mask,
            "stage1_headwear_mask": stage1_headwear_mask,
            "metadata": s1_metadata,
            "warnings": warnings,
            "experiment_dir": exp_dir_path,
        }

    # =========================================================
    # STAGE 2: REFERENCE HAIRSTYLE TRANSFER ONTO BALD BASE
    # =========================================================
    logger.info("TwoStageTransfer: Starting Stage 2 (Reference Hairstyle Transfer onto Bald Base)...")
    s2_t0 = time.perf_counter()

    # Reference background cleanup
    ref_rgb_np = np.asarray(ref_img_rgb)
    ref_hair_np = np.asarray(ref_hair_mask_clean)
    cleaned_ref_hair_np, prune_info = refine_reference_hair_mask(
        ref_rgb_np,
        ref_hair_np,
        landmarks_dict=reference_landmarks,
        reference_segmentation_map=reference_segmentation_map,
    )
    cleaned_ref_hair_pil = Image.fromarray(cleaned_ref_hair_np, mode="L")
    pruned_count = int(prune_info.get("pruned_count", 0))
    if pruned_count > 0:
        warnings.append(f"Stage 2: Pruned {pruned_count} false-positive backdrop pixels from reference hair mask.")

    # IP-Adapter reference crop
    # IP-Adapter reference crop
    ref_crop = prepare_reference_crop(ref_img_rgb, cleaned_ref_hair_pil)

    # Stage 2 residual hair detection on bald_base (Requirement A)
    # Do not silently assume residual mask is empty. Segment actual bald base:
    from src.hair_segmentation import segment_hair as app_segment_hair

    bald_hair_mask_pil, _, _, _, bald_seg_map = app_segment_hair(bald_base, return_seg_map=True)
    bald_hair_np = np.asarray(bald_hair_mask_pil.convert("L"))
    residual_hair_count = int(np.count_nonzero(bald_hair_np > 128))

    # If genuine residual hair was detected (> 500 px), mark for repair; otherwise clean zero mask
    if residual_hair_count > 500:
        residual_hair_mask_pil = bald_hair_mask_pil
        warnings.append(
            f"Stage 2: Detected {residual_hair_count} residual hair pixels on bald base; marking for repair."
        )
    else:
        residual_hair_mask_pil = Image.new("L", (w_t, h_t), 0)

    # Hairstyle alignment:
    # Target image for alignment is the STAGE 1 BALD BASE!
    # Target hair mask passed is the RESIDUAL hair mask (ensures clean bald scalp is preserved
    # and prevents false pompadour removal in empty space)
    alignment_result = align_reference_to_target(
        target_image=bald_base,
        reference_image=ref_img_rgb,
        target_hair_mask=residual_hair_mask_pil,
        reference_hair_mask=cleaned_ref_hair_pil,
        target_landmarks=face_landmarks,
        reference_landmarks=reference_landmarks,
        target_segmentation_map=bald_seg_map,
        reference_segmentation_map=reference_segmentation_map,
        cranial_expansion_px=final_cranial,
        feather_width_px=final_feather,
        hairline_transition_px=final_transition,
        donor_coverage_threshold=final_threshold,
        contact_shadow_strength=s2_cfg.get("contact_shadow_strength", 0.25),
    )

    if alignment_result.get("warnings"):
        warnings.extend(alignment_result["warnings"])

    aligned_donor_rgb = alignment_result["aligned_hair_rgb"]
    aligned_donor_alpha = alignment_result["aligned_hair_alpha"]
    aligned_composite_before_diffusion = alignment_result["pasted_composite"]
    edge_repair_mask = alignment_result["edge_repair_mask"]
    broad_edit_mask = alignment_result["broad_edit_mask"]
    identity_protection = alignment_result.get("identity_protection")
    contact_shadow_map = alignment_result.get("contact_shadow_map")
    validation_result = alignment_result.get("validation_result", {})
    validation_overlay = alignment_result.get("validation_overlay")

    # Inpainting on bald base pre-composite with aspect-preserving dimensions divisible by 8
    inpainting_base = aligned_composite_before_diffusion
    edge_mask_np = np.asarray(edge_repair_mask)
    active_repair_px = int(np.count_nonzero(edge_mask_np > 128))

    # Evaluate whether seam diffusion is necessary and valid (Requirement B & E)
    should_inpaint = validation_result.get("should_inpaint", active_repair_px > 0)
    if active_repair_px == 0:
        should_inpaint = False
        logger.info("TwoStageTransfer: Edge repair mask is empty (0 px). Aligned pre-diffusion composite is preserved.")
        warnings.append("Stage 2: Hairline seam is already photorealistic; diffusion step safely bypassed.")
    elif not validation_result.get("is_valid", True):
        logger.warning(
            f"TwoStageTransfer: Repair mask validation flagged issues: {validation_result.get('reasons')}. Using sanitized mask."
        )
        warnings.append(f"Stage 2: Sanitized edge repair mask applied: {validation_result.get('reasons')}")
        edge_mask_np = validation_result.get("sanitized_mask_np", edge_mask_np)
        edge_repair_mask = Image.fromarray(edge_mask_np, mode="L")

    stage2_seed = final_seed + 1000

    stage2_raw_generated_crop = None
    stage2_hairline_roi = None
    stage2_hairline_roi_metadata = {}
    stage2_seam_blend_preview = None
    compare_prediff_vs_final_forehead = None

    if should_inpaint:
        # 1. Derive padded ROI crop around genuine hairline seam
        y_idx, x_idx = np.where(edge_mask_np > 128)
        y_min, y_max = int(y_idx.min()), int(y_idx.max())
        x_min, x_max = int(x_idx.min()), int(x_idx.max())

        geom_t = compute_face_geometry(face_landmarks, w_t, h_t)
        face_scale = geom_t["face_scale"]

        # Scale-adaptive padding around the seam to provide forehead and hair context
        pad_y = int(np.clip(face_scale * 0.15, 48, 128))
        pad_x = int(np.clip(face_scale * 0.12, 48, 128))

        crop_y1 = max(0, y_min - pad_y)
        crop_y2 = min(h_t, y_max + pad_y)
        crop_x1 = max(0, x_min - pad_x)
        crop_x2 = min(w_t, x_max + pad_x)

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        roi_img = inpainting_base.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        roi_mask = edge_repair_mask.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # Visual ROI bounding box preview
        roi_preview_np = np.asarray(inpainting_base).copy()
        cv2.rectangle(roi_preview_np, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 4)
        stage2_hairline_roi = Image.fromarray(roi_preview_np, mode="RGB")

        # 2. Aspect-ratio preserving dimensions divisible by 8 for model (native pixel resolution)
        inpaint_w, inpaint_h = compute_inpaint_dimensions(crop_w, crop_h, max_dim=768, min_dim=512)
        inpaint_input = roi_img.resize((inpaint_w, inpaint_h), Image.Resampling.LANCZOS)
        inpaint_mask = roi_mask.resize((inpaint_w, inpaint_h), Image.Resampling.NEAREST)

        stage2_hairline_roi_metadata = {
            "crop_box": [crop_x1, crop_y1, crop_x2, crop_y2],
            "crop_dimensions": [crop_w, crop_h],
            "inpaint_dimensions": [inpaint_w, inpaint_h],
            "edge_repair_pixels": active_repair_px,
            "scale_x": float(inpaint_w / crop_w),
            "scale_y": float(inpaint_h / crop_h),
        }

        # Configure diffusion generator with Stage 2 seed (derived deterministically)
        import torch

        device = "cpu"
        if hasattr(shared_pipeline, "device") and isinstance(getattr(shared_pipeline, "device"), (str, torch.device)):
            device = shared_pipeline.device
        generator = torch.Generator(device=device).manual_seed(stage2_seed)

        if hasattr(shared_pipeline, "set_ip_adapter_scale"):
            shared_pipeline.set_ip_adapter_scale(final_scale)

        donor_core_px = aligned_donor_rgb[aligned_donor_alpha > 0.7]
        color_desc = describe_hair_color(donor_core_px)
        color_term = f"{color_desc} " if color_desc else ""
        prompt = (
            f"natural realistic {color_term}hair, highly detailed, clean hairline, "
            "high resolution photography, hair color exactly matching the surrounding hair, "
            "consistent color, no color shift"
        )
        negative_prompt = (
            "white patch, bright white spot, washed out highlight, overexposed patch, flat color block, "
            "visible seam, hard rectangular edge, purple, violet, cyan, chromatic aberration, mesh, grid, "
            "dithering, noise, pixelation, bald spots, thin hair, gaps, blur, artifacts, plastic, drawing, cartoon, bad anatomy"
        )

        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": inpaint_input,
            "mask_image": inpaint_mask,
            "num_inference_steps": final_s2_steps,
            "guidance_scale": final_guidance,
            "generator": generator,
        }
        if hasattr(shared_pipeline, "image_encoder") and shared_pipeline.image_encoder is not None:
            pipe_kwargs["ip_adapter_image"] = ref_crop

        diffusion_output = shared_pipeline(**pipe_kwargs)
        stage2_raw_generated_crop = diffusion_output.images[0]

        # Resize crop back to original crop dimensions with LANCZOS
        raw_crop_restored = stage2_raw_generated_crop.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
        raw_crop_restored_np = np.asarray(raw_crop_restored.convert("RGB"))
        base_np_full = np.asarray(inpainting_base.convert("RGB"))

        clone_mask_crop = np.asarray(roi_mask.convert("L"))
        clone_mask_crop = (clone_mask_crop > 20).astype(np.uint8) * 255

        stage2_raw_generated = inpainting_base.copy()
        stage2_raw_generated.paste(raw_crop_restored, (crop_x1, crop_y1))  # safe fallback, overwritten below if clone succeeds

        if np.count_nonzero(clone_mask_crop) > 50:
            ys_m, xs_m = np.where(clone_mask_crop > 0)
            # seamlessClone requires the mask to not touch the patch border
            touches_border = (
                ys_m.min() <= 1 or xs_m.min() <= 1
                or ys_m.max() >= clone_mask_crop.shape[0] - 2
                or xs_m.max() >= clone_mask_crop.shape[1] - 2
            )
            if not touches_border:
                center = (
                    int(crop_x1 + (xs_m.min() + xs_m.max()) / 2),
                    int(crop_y1 + (ys_m.min() + ys_m.max()) / 2),
                )
                try:
                    cloned_np = cv2.seamlessClone(
                        raw_crop_restored_np, base_np_full, clone_mask_crop, center, cv2.NORMAL_CLONE,
                    )
                    stage2_raw_generated = Image.fromarray(cloned_np)
                except cv2.error as e:
                    logger.warning(f"seamlessClone failed, using flat paste fallback: {e}")
    else:
        stage2_raw_generated = inpainting_base
        stage2_hairline_roi = inpainting_base.copy()
        stage2_hairline_roi_metadata = {"status": "bypassed_or_empty"}

    # Final Compositing & Strict Identity / Outside-Region Preservation
    # 1. Start with inpainting base
    raw_np = np.asarray(stage2_raw_generated)
    inpaint_base_np = np.asarray(inpainting_base)
    broad_mask_np = np.asarray(broad_edit_mask)
    orig_target_np = np.asarray(target_img_rgb)

    if should_inpaint:
        # Feather edge repair mask for seamless blending
        feather_repair = max(1.0, float(final_feather))
        dist_repair = cv2.distanceTransform((edge_mask_np > 128).astype(np.uint8), cv2.DIST_L2, 5)
        blend_repair = np.clip(dist_repair / feather_repair, 0.0, 1.0).astype(np.float32)
        blend_repair[edge_mask_np == 0] = 0.0

        # Composite diffusion into inpainting base strictly inside edge repair mask
        final_np = (
            inpaint_base_np.astype(np.float32) * (1.0 - blend_repair[..., None])
            + raw_np.astype(np.float32) * blend_repair[..., None]
        ).clip(0, 255).astype(np.uint8)

        # Diagnostics: seam blend preview & forehead zoom
        stage2_seam_blend_preview = Image.fromarray((blend_repair * 255).astype(np.uint8), mode="L")

        y_zoom1 = max(0, int(crop_y1 + crop_h * 0.1))
        y_zoom2 = min(h_t, int(crop_y1 + crop_h * 0.9))
        x_zoom1 = max(0, int(crop_x1 + crop_w * 0.1))
        x_zoom2 = min(w_t, int(crop_x1 + crop_w * 0.9))
        crop_before_fh = inpaint_base_np[y_zoom1:y_zoom2, x_zoom1:x_zoom2]
        crop_after_fh = final_np[y_zoom1:y_zoom2, x_zoom1:x_zoom2]
        forehead_sbs = np.hstack([crop_before_fh, crop_after_fh])
        compare_prediff_vs_final_forehead = Image.fromarray(forehead_sbs, mode="RGB")
    else:
        final_np = inpaint_base_np.copy()
        stage2_seam_blend_preview = Image.new("L", (w_t, h_t), 0)
        compare_prediff_vs_final_forehead = inpainting_base.copy()

    # 2. Strictly protect facial features (eyes, eyebrows, nose, mouth) while allowing donor hair
    # to naturally overlap ear cartilage and drape over shoulders as established in baseline
    if identity_protection is not None:
        ident_np = np.asarray(identity_protection)
        if ident_np.shape[:2] != (h_t, w_t):
            ident_np = cv2.resize(ident_np, (w_t, h_t), interpolation=cv2.INTER_NEAREST)

        face_core_protection = ident_np.copy()

        # Determine base reference for restoring facial features:
        # If a pixel was modified during Stage 1 (under old hair mask, e.g. ears/sideburns/scalp revealed),
        # bald_base contains the clean reconstructed skin/ear rather than orig_target_np which might contain
        # old hair or background wall cutouts.
        s1_mask_np = np.asarray(stage1_edit_mask)
        if s1_mask_np.shape[:2] != (h_t, w_t):
            s1_mask_np = cv2.resize(s1_mask_np, (w_t, h_t), interpolation=cv2.INTER_NEAREST)
        bald_np_aligned = np.asarray(bald_base.convert("RGB"))
        if bald_np_aligned.shape[:2] != (h_t, w_t):
            bald_np_aligned = cv2.resize(bald_np_aligned, (w_t, h_t), interpolation=cv2.INTER_LINEAR)

        base_ref_np = np.where(s1_mask_np[..., None] > 128, bald_np_aligned, orig_target_np)

        if bald_seg_map is not None:
            ear_cartilage_mask = np.isin(bald_seg_map, EAR_SEG_CLASSES)
            ear_alpha = np.clip(aligned_donor_alpha, 0.0, 1.0)

            # Only genuinely opaque strands are allowed to fully override ear
            # protection. Below this, we're looking at thin/translucent edge
            # pixels that are mostly NOT hair -- letting those win at a flat
            # 0.25 cutoff exposes bald-base/background color right at the ear,
            # which is exactly the white-patch artifact.
            EAR_HAIR_OPAQUE = 0.55

            hard_hair_over_ears = ear_cartilage_mask & (ear_alpha >= EAR_HAIR_OPAQUE)
            face_core_protection[hard_hair_over_ears] = 0

            # Ambiguous band (some but not full coverage): alpha-blend against
            # the REAL original ear pixels (or bald_base if ear was revealed in Stage 1),
            # so partial coverage reads as genuine fine strands over real skin instead of a gap.
            soft_ear_band = ear_cartilage_mask & (ear_alpha > 0.0) & (ear_alpha < EAR_HAIR_OPAQUE)
            if np.any(soft_ear_band):
                blend_w = (ear_alpha[soft_ear_band] / EAR_HAIR_OPAQUE)[..., None]
                final_np[soft_ear_band] = (
                    base_ref_np[soft_ear_band].astype(np.float32) * (1.0 - blend_w)
                    + final_np[soft_ear_band].astype(np.float32) * blend_w
                ).clip(0, 255).astype(np.uint8)
                face_core_protection[soft_ear_band] = 0  # already handled, skip the hard restore below

        final_np[face_core_protection > 0] = base_ref_np[face_core_protection > 0]

    # 3. Restore ORIGINAL target pixels outside combined broad replacement envelope
    # Ensures the broad replacement envelope includes the union of permitted changes introduced
    # by BOTH Stage 1 (bald/scalp creation) and Stage 2 (hair transfer/blending).
    # Otherwise, old target hair cleared in Stage 1 could be restored over the final result.
    if 's1_mask_np' not in locals():
        s1_mask_np = np.asarray(stage1_edit_mask)
        if s1_mask_np.shape[:2] != (h_t, w_t):
            s1_mask_np = cv2.resize(s1_mask_np, (w_t, h_t), interpolation=cv2.INTER_NEAREST)
    combined_envelope_np = np.maximum(broad_mask_np, s1_mask_np)

    outside_envelope = combined_envelope_np == 0
    final_np[outside_envelope] = orig_target_np[outside_envelope]

    final_composite = Image.fromarray(final_np, mode="RGB")
    s2_duration = time.perf_counter() - s2_t0
    total_duration = time.perf_counter() - t_start

    # Compute Metrics
    has_nan_inf = bool(np.isnan(final_np).any() or np.isinf(final_np).any())
    is_blank = bool(final_np.std() < 1e-2)
    preserv_metrics = compute_preservation_metrics(
        original_target_np=orig_target_np,
        final_composite_np=final_np,
        broad_envelope_np=combined_envelope_np,
        identity_protection_np=identity_protection,
    )

    metrics = {
        "dimension_match": final_composite.size == (w_t, h_t),
        "outside_pixel_preservation_pct": preserv_metrics["outside_pixel_preservation_pct"],
        "max_outside_pixel_diff": preserv_metrics["max_outside_pixel_diff"],
        "mean_outside_pixel_diff": preserv_metrics["mean_outside_pixel_diff"],
        "identity_protection_leak_count": preserv_metrics["identity_protection_leak_count"],
        "stage1_edit_mask_coverage_pct": round(
            (np.count_nonzero(s1_mask_np > 128) / (w_t * h_t)) * 100.0, 2
        ),
        "stage2_edit_mask_coverage_pct": round(
            (np.count_nonzero(edge_mask_np > 128) / (w_t * h_t)) * 100.0, 2
        ),
        "broad_envelope_coverage_pct": round(
            (np.count_nonzero(combined_envelope_np > 128) / (w_t * h_t)) * 100.0, 2
        ),
        "donor_core_pixel_count": int(np.count_nonzero(aligned_donor_alpha > 0.65)),
        "has_nan_or_inf": has_nan_inf,
        "is_blank_image": is_blank,
        "stage1_duration_seconds": round(s1_duration, 2),
        "stage2_duration_seconds": round(s2_duration, 2),
        "total_duration_seconds": round(total_duration, 2),
        "peak_memory_mb": round(get_process_memory_mb(), 1),
    }

    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_dimensions": [w_t, h_t],
        "reference_dimensions": [w_r, h_r],
        "stage1_backend": s1_result.metadata.get("backend", final_s1_backend) if ('s1_result' in locals() and s1_result is not None) else "precomputed",
        "stage1_steps": final_s1_steps,
        "stage1_seed": final_seed,
        "stage2_steps": final_s2_steps,
        "stage2_seed": stage2_seed,
        "ip_adapter_scale": final_scale,
        "guidance_scale": final_guidance,
        "controlnet_active": controlnet_active,
        "controlnet_reason": controlnet_reason,
        "donor_coverage_threshold": final_threshold,
    }

    # Record transform and pose diagnostics in metadata
    serializable_transform = {}
    if "transform_info" in alignment_result and isinstance(alignment_result["transform_info"], dict):
        for k, v in alignment_result["transform_info"].items():
            if isinstance(v, np.ndarray):
                serializable_transform[k] = v.tolist()
            elif isinstance(v, (np.floating, float)):
                serializable_transform[k] = round(float(v), 4)
            elif isinstance(v, (np.integer, int, bool, str)):
                serializable_transform[k] = v
            else:
                serializable_transform[k] = str(v)
    metadata["transform_info"] = serializable_transform

    # Save Experiment Artifacts if requested
    exp_dir_path: Optional[str] = None
    if save_experiment:
        run_tag = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir:
            exp_dir = Path(output_dir) / f"{run_tag}"
        else:
            exp_dir = Path("outputs/two_stage_runs") / f"{run_tag}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_dir_path = str(exp_dir.resolve())

        # Save images
        target_img_rgb.save(exp_dir / "input_target.png")
        ref_img_rgb.save(exp_dir / "input_reference.png")
        target_hair_mask_clean.save(exp_dir / "target_hair_mask.png")
        ref_hair_mask_clean.save(exp_dir / "reference_hair_mask_original.png")
        cleaned_ref_hair_pil.save(exp_dir / "reference_hair_mask_cleaned.png")

        # Reference mask overlay
        ref_overlay = ref_rgb_np.copy()
        mask_b = cleaned_ref_hair_np > 128
        Image.fromarray(ref_overlay).save(exp_dir / "reference_mask_overlay.png")

        # Selective fill diagnostic masks
        if alignment_result.get("reference_mask_filled") is not None:
            Image.fromarray(alignment_result["reference_mask_filled"]).save(exp_dir / "reference_hair_mask_filled.png")
        if alignment_result.get("reference_fill_mask") is not None:
            Image.fromarray(alignment_result["reference_fill_mask"]).save(exp_dir / "reference_hair_mask_holes.png")

        # Alignment anchor diagnostic overlay
        if (face_landmarks and face_landmarks.get("detected") and
            reference_landmarks and reference_landmarks.get("detected") and
            alignment_result.get("transform_info", {}).get("matrix") is not None):
            try:
                from src.hairstyle_alignment import ALIGNMENT_ANCHOR_INDICES, build_symmetric_correspondence
                t_diag = orig_target_np.copy()
                M = alignment_result["transform_info"]["matrix"]
                src_pts, dst_pts = build_symmetric_correspondence(
                    reference_landmarks["landmarks_2d"],
                    face_landmarks["landmarks_2d"],
                    is_src_mirrored=alignment_result.get("pose_info", {}).get("recommend_mirror", False),
                    anchor_indices=ALIGNMENT_ANCHOR_INDICES,
                )
                ones = np.ones((len(src_pts), 1))
                warped_src = (np.hstack([src_pts, ones]) @ M.T).astype(np.int32)
                for p_dst, p_warp in zip(dst_pts.astype(np.int32), warped_src):
                    cv2.circle(t_diag, tuple(p_dst), 10, (0, 255, 255), -1)  # Yellow: Target anchors
                    cv2.circle(t_diag, tuple(p_warp), 8, (255, 0, 255), -1)  # Magenta: Warped Ref anchors
                    cv2.line(t_diag, tuple(p_dst), tuple(p_warp), (0, 255, 0), 2)  # Green: Residual
                Image.fromarray(t_diag).save(exp_dir / "alignment_anchor_diagnostic.png")
            except Exception as e:
                logger.warning(f"Failed to generate alignment anchor diagnostic: {e}")

        stage1_edit_mask.save(exp_dir / "stage1_edit_mask.png")
        stage1_protection_mask.save(exp_dir / "stage1_protection_mask.png")
        if stage1_removal_overlay:
            stage1_removal_overlay.save(exp_dir / "stage1_removal_overlay.png")
        if stage1_headwear_mask:
            stage1_headwear_mask.save(exp_dir / "stage1_headwear_mask.png")
        if stage1_raw_generated:
            stage1_raw_generated.save(exp_dir / "stage1_raw_generated.png")
        bald_base.save(exp_dir / "bald_base.png")

        # Phase 1: Background Repair Diagnostics
        if stage1_bg_repair_mask is not None:
            stage1_bg_repair_mask.save(exp_dir / "stage1_background_repair_mask.png")
        if stage1_bg_repair_preview is not None:
            stage1_bg_repair_preview.save(exp_dir / "stage1_background_repair_preview.png")
        if stage1_bald_base_before_after is not None:
            stage1_bald_base_before_after.save(exp_dir / "stage1_bald_base_before_after.png")
        if stage1_bg_difference is not None:
            stage1_bg_difference.save(exp_dir / "stage1_background_difference.png")

        # Phase 2: Donor Skin Exclusion & Edge Refinement Diagnostics
        ref_skin_excl = alignment_result.get("reference_skin_exclusion_mask")
        if ref_skin_excl is not None:
            ref_skin_excl.save(exp_dir / "reference_skin_exclusion_mask.png")
        ref_alpha_refined = alignment_result.get("reference_hair_alpha_refined")
        if ref_alpha_refined is not None:
            ref_alpha_refined.save(exp_dir / "reference_hair_alpha_refined.png")
        ref_mask_sbs = alignment_result.get("reference_hair_mask_before_after")
        if ref_mask_sbs is not None:
            ref_mask_sbs.save(exp_dir / "reference_hair_mask_before_after.png")
        donor_bleed_prev = alignment_result.get("donor_skin_bleed_preview")
        if donor_bleed_prev is not None:
            donor_bleed_prev.save(exp_dir / "donor_skin_bleed_preview.png")

        # Aligned donor RGB & Alpha
        Image.fromarray(aligned_donor_rgb).save(exp_dir / "aligned_donor_rgb.png")
        Image.fromarray((np.clip(aligned_donor_alpha, 0, 1) * 255).astype(np.uint8)).save(
            exp_dir / "aligned_donor_alpha.png"
        )
        ref_crop.save(exp_dir / "ip_adapter_reference_crop.png")

        broad_edit_mask.save(exp_dir / "stage2_broad_edit_mask.png")
        edge_repair_mask.save(exp_dir / "stage2_edge_repair_mask.png")
        residual_hair_mask_pil.save(exp_dir / "residual_hair_mask.png")
        if validation_overlay is not None:
            Image.fromarray(validation_overlay).save(exp_dir / "repair_mask_validation_overlay.png")
        if contact_shadow_map is not None and np.any(contact_shadow_map > 0):
            Image.fromarray((np.clip(contact_shadow_map, 0, 1) * 255).astype(np.uint8)).save(
                exp_dir / "contact_shadow_map.png"
            )
        aligned_composite_before_diffusion.save(exp_dir / "aligned_composite_before_diffusion.png")
        stage2_raw_generated.save(exp_dir / "stage2_raw_generated.png")

        # Phase 3: Localized Hairline ROI Diagnostics
        if stage2_hairline_roi is not None:
            stage2_hairline_roi.save(exp_dir / "stage2_hairline_roi.png")
        with (exp_dir / "stage2_hairline_roi_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(stage2_hairline_roi_metadata, f, indent=2)
        if stage2_raw_generated_crop is not None:
            stage2_raw_generated_crop.save(exp_dir / "stage2_raw_generated_crop.png")
        if stage2_seam_blend_preview is not None:
            stage2_seam_blend_preview.save(exp_dir / "stage2_seam_blend_preview.png")
        if compare_prediff_vs_final_forehead is not None:
            compare_prediff_vs_final_forehead.save(exp_dir / "compare_prediff_vs_final_forehead.png")

        # 5-Panel Comparison Collage:
        # 1. Original Target | 2. Reference | 3. Bald Base | 4. Pre-Diffusion Composite | 5. Final Output
        ref_resized = ref_img_rgb.resize((w_t, h_t), Image.Resampling.LANCZOS)
        collage = np.hstack([
            orig_target_np,
            np.asarray(ref_resized),
            np.asarray(bald_base),
            np.asarray(aligned_composite_before_diffusion),
            final_np,
        ])
        Image.fromarray(collage).save(exp_dir / "comparison_collage.png")

        # Crop back to original dimensions for final artifacts
        if pad_top_px > 0:
            final_composite = final_composite.crop((0, pad_top_px, orig_w_t, h_t))
            bald_base = bald_base.crop((0, pad_top_px, orig_w_t, h_t))

        final_composite.save(exp_dir / "final_composite.png")
        bald_base.save(exp_dir / "bald_base.png")

        # JSON Metadata & Metrics
        with (exp_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        with (exp_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with (exp_dir / "warnings.json").open("w", encoding="utf-8") as f:
            json.dump(warnings, f, indent=2)

        logger.info(f"TwoStageTransfer: All artifacts successfully saved to {exp_dir}")
    else:
        if pad_top_px > 0:
            final_composite = final_composite.crop((0, pad_top_px, orig_w_t, h_t))
            bald_base = bald_base.crop((0, pad_top_px, orig_w_t, h_t))

    return {
        "composite_image": final_composite,
        "final_composite": final_composite,
        "bald_base": bald_base,
        "stage1_raw_generated": stage1_raw_generated,
        "stage1_edit_mask": stage1_edit_mask,
        "stage1_background_repair_mask": stage1_bg_repair_mask,
        "stage1_background_repair_preview": stage1_bg_repair_preview,
        "stage1_bald_base_before_after": stage1_bald_base_before_after,
        "stage1_background_difference": stage1_bg_difference,
        "reference_skin_exclusion_mask": alignment_result.get("reference_skin_exclusion_mask"),
        "reference_hair_alpha_refined": alignment_result.get("reference_hair_alpha_refined"),
        "reference_hair_mask_before_after": alignment_result.get("reference_hair_mask_before_after"),
        "donor_skin_bleed_preview": alignment_result.get("donor_skin_bleed_preview"),
        "stage2_raw_generated": stage2_raw_generated,
        "stage2_raw_generated_crop": stage2_raw_generated_crop,
        "stage2_hairline_roi": stage2_hairline_roi,
        "stage2_hairline_roi_metadata": stage2_hairline_roi_metadata,
        "stage2_seam_blend_preview": stage2_seam_blend_preview,
        "compare_prediff_vs_final_forehead": compare_prediff_vs_final_forehead,
        "edge_repair_mask": edge_repair_mask,
        "residual_hair_mask": residual_hair_mask_pil,
        "broad_edit_mask": broad_edit_mask,
        "aligned_composite_before_diffusion": aligned_composite_before_diffusion,
        "aligned_donor_rgb": aligned_donor_rgb,
        "aligned_donor_alpha": aligned_donor_alpha,
        "contact_shadow_map": contact_shadow_map,
        "validation_result": validation_result,
        "validation_overlay": validation_overlay,
        "reference_crop": ref_crop,
        "metrics": metrics,
        "metadata": metadata,
        "warnings": warnings,
        "experiment_dir": exp_dir_path,
    }
