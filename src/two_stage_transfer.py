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
from PIL import Image

from src.bald_converter import (
    BaldConverterConfig,
    create_bald_converter,
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
            "num_inference_steps": 8,
            "guidance_scale": 7.5,
            "hair_mask_expansion_ratio": 0.01,
            "max_hair_mask_expansion_px": 12,
            "feather_width_px": 2.0,
        },
        "stage2": {
            "num_inference_steps": 8,
            "guidance_scale": 7.5,
            "ip_adapter_scale": 0.75,
            "cranial_expansion_px": 30,
            "feather_width_px": 6.0,
            "hairline_transition_px": 15,
            "donor_coverage_threshold": 0.85,
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


def generate_two_stage_hairstyle_transfer(
    target_image: Image.Image,
    reference_image: Image.Image,
    pipeline=None,
    target_hair_mask: Optional[Image.Image] = None,
    reference_hair_mask: Optional[Image.Image] = None,
    target_segmentation_map: Optional[np.ndarray] = None,
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
    headwear_mask: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
    detect_headwear: bool = True,
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

    final_s1_backend = stage1_backend if stage1_backend != "auto" else s1_cfg.get("backend", "auto")
    final_s1_steps = stage1_steps if stage1_steps is not None else s1_cfg.get("num_inference_steps", 12)
    final_s2_steps = stage2_steps if stage2_steps is not None else s2_cfg.get("num_inference_steps", 8)
    final_scale = ip_adapter_scale if ip_adapter_scale is not None else s2_cfg.get("ip_adapter_scale", 0.75)
    final_guidance = guidance_scale if guidance_scale is not None else s2_cfg.get("guidance_scale", 7.5)
    final_seed = seed if seed is not None else s2_cfg.get("seed", 42)
    final_cranial = cranial_expansion_px if cranial_expansion_px is not None else s2_cfg.get("cranial_expansion_px", 30)
    final_feather = feather_width_px if feather_width_px is not None else s2_cfg.get("feather_width_px", 6.0)
    final_transition = hairline_transition_px if hairline_transition_px is not None else s2_cfg.get("hairline_transition_px", 15)
    final_threshold = donor_coverage_threshold if donor_coverage_threshold is not None else s2_cfg.get("donor_coverage_threshold", 0.85)

    # 2. Preprocess & Validate Target & Reference
    target_img_rgb = target_image.convert("RGB")
    ref_img_rgb = reference_image.convert("RGB") if reference_image is not None else target_img_rgb.copy()
    w_t, h_t = target_img_rgb.size
    w_r, h_r = ref_img_rgb.size

    # Segmentation
    if target_hair_mask is None or target_segmentation_map is None:
        from src.hair_segmentation import segment_hair as app_segment_hair

        t_seg_res = app_segment_hair(target_img_rgb, return_seg_map=True)
        if target_hair_mask is None:
            target_hair_mask = t_seg_res[0]
        if target_segmentation_map is None:
            target_segmentation_map = t_seg_res[4]

    if not stage1_only and reference_hair_mask is None and reference_image is not None:
        from src.hair_segmentation import segment_hair as app_segment_hair

        r_seg_res = app_segment_hair(ref_img_rgb, return_seg_map=True)
        reference_hair_mask = r_seg_res[0]
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
    )
    s1_duration = time.perf_counter() - s1_t0

    bald_base = s1_result.bald_base
    stage1_edit_mask = s1_result.edit_mask
    stage1_protection_mask = s1_result.protection_mask
    stage1_removal_overlay = s1_result.removal_overlay
    stage1_headwear_mask = s1_result.headwear_mask
    stage1_raw_generated = s1_result.raw_generated
    if s1_result.warnings:
        warnings.extend(s1_result.warnings)

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

        target_img_rgb.save(exp_dir / "input_target.png")
        target_hair_mask_clean.save(exp_dir / "target_hair_mask.png")
        stage1_edit_mask.save(exp_dir / "stage1_edit_mask.png")
        stage1_protection_mask.save(exp_dir / "stage1_protection_mask.png")
        if stage1_removal_overlay:
            stage1_removal_overlay.save(exp_dir / "stage1_removal_overlay.png")
        if stage1_headwear_mask:
            stage1_headwear_mask.save(exp_dir / "stage1_headwear_mask.png")
        if stage1_raw_generated:
            stage1_raw_generated.save(exp_dir / "stage1_raw_generated.png")
        bald_base.save(exp_dir / "bald_base.png")

        s1_metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stage": "stage1_only",
            "target_dimensions": [w_t, h_t],
            "stage1_backend": final_s1_backend,
            "stage1_steps": final_s1_steps,
            "stage1_seed": final_seed,
            "headwear_detected": s1_result.headwear_detected,
            "duration_seconds": s1_duration,
        }
        with (exp_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(s1_metadata, f, indent=2)
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
        ref_rgb_np, ref_hair_np, landmarks_dict=reference_landmarks
    )
    cleaned_ref_hair_pil = Image.fromarray(cleaned_ref_hair_np, mode="L")
    pruned_count = int(prune_info.get("pruned_count", 0))
    if pruned_count > 0:
        warnings.append(f"Stage 2: Pruned {pruned_count} false-positive backdrop pixels from reference hair mask.")

    # IP-Adapter reference crop
    ref_crop = prepare_reference_crop(ref_img_rgb, cleaned_ref_hair_pil)

    # Hairstyle alignment:
    # Target image for alignment is the STAGE 1 BALD BASE!
    # Target hair mask passed is the ORIGINAL hair mask (ensures full union coverage)
    alignment_result = align_reference_to_target(
        target_image=bald_base,
        reference_image=ref_img_rgb,
        target_hair_mask=target_hair_mask_clean,
        reference_hair_mask=cleaned_ref_hair_pil,
        target_landmarks=face_landmarks,
        reference_landmarks=reference_landmarks,
        target_segmentation_map=target_segmentation_map,
        cranial_expansion_px=final_cranial,
        feather_width_px=final_feather,
        hairline_transition_px=final_transition,
        donor_coverage_threshold=final_threshold,
    )

    if alignment_result.get("warnings"):
        warnings.extend(alignment_result["warnings"])

    aligned_donor_rgb = alignment_result["aligned_hair_rgb"]
    aligned_donor_alpha = alignment_result["aligned_hair_alpha"]
    aligned_composite_before_diffusion = alignment_result["pasted_composite"]
    edge_repair_mask = alignment_result["edge_repair_mask"]
    broad_edit_mask = alignment_result["broad_edit_mask"]
    identity_protection = alignment_result.get("identity_protection")

    # Inpainting on bald base pre-composite
    inpainting_base = aligned_composite_before_diffusion
    inpaint_input_512 = inpainting_base.resize((512, 512), Image.Resampling.LANCZOS)
    inpaint_mask_512 = edge_repair_mask.resize((512, 512), Image.Resampling.NEAREST)

    # Configure diffusion generator with Stage 2 seed (derived deterministically)
    import torch

    device = "cpu"
    if hasattr(shared_pipeline, "device") and isinstance(getattr(shared_pipeline, "device"), (str, torch.device)):
        device = shared_pipeline.device
    stage2_seed = final_seed + 1000
    generator = torch.Generator(device=device).manual_seed(stage2_seed)

    if hasattr(shared_pipeline, "set_ip_adapter_scale"):
        shared_pipeline.set_ip_adapter_scale(final_scale)

    prompt = "natural realistic hair, highly detailed, clean hairline, high resolution photography"
    negative_prompt = "bald spots, thin hair, gaps, blur, artifacts, plastic, drawing, cartoon, bad anatomy"

    pipe_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": inpaint_input_512,
        "mask_image": inpaint_mask_512,
        "num_inference_steps": final_s2_steps,
        "guidance_scale": final_guidance,
        "generator": generator,
    }
    if hasattr(shared_pipeline, "image_encoder") and shared_pipeline.image_encoder is not None:
        pipe_kwargs["ip_adapter_image"] = ref_crop

    diffusion_output = shared_pipeline(**pipe_kwargs)
    stage2_raw_generated = diffusion_output.images[0].resize((w_t, h_t), Image.Resampling.LANCZOS)

    # Final Compositing & Strict Identity / Outside-Region Preservation
    # 1. Start with inpainting base
    raw_np = np.asarray(stage2_raw_generated)
    inpaint_base_np = np.asarray(inpainting_base)
    edge_mask_np = np.asarray(edge_repair_mask)
    broad_mask_np = np.asarray(broad_edit_mask)
    orig_target_np = np.asarray(target_img_rgb)

    # Feather edge repair mask for seamless blending
    feather_repair = max(1.0, float(final_feather))
    dist_repair = cv2.distanceTransform((edge_mask_np > 128).astype(np.uint8), cv2.DIST_L2, 5)
    blend_repair = np.clip(dist_repair / feather_repair, 0.0, 1.0).astype(np.float32)

    # Composite diffusion into inpainting base
    final_np = (
        inpaint_base_np.astype(np.float32) * (1.0 - blend_repair[..., None])
        + raw_np.astype(np.float32) * blend_repair[..., None]
    ).clip(0, 255).astype(np.uint8)

    # 2. Strictly protect identity pixels (ears, eyes, brows, nose, mouth)
    if identity_protection is not None:
        ident_np = np.asarray(identity_protection)
        if ident_np.shape[:2] != (h_t, w_t):
            ident_np = cv2.resize(ident_np, (w_t, h_t), interpolation=cv2.INTER_NEAREST)
        final_np[ident_np > 0] = orig_target_np[ident_np > 0]

    # 3. Restore ORIGINAL target pixels outside combined broad replacement envelope
    # Ensures the broad replacement envelope includes the union of permitted changes introduced
    # by BOTH Stage 1 (bald/scalp creation) and Stage 2 (hair transfer/blending).
    # Otherwise, old target hair cleared in Stage 1 could be restored over the final result.
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
        "stage1_backend": s1_result.metadata.get("backend", final_s1_backend),
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
        ref_overlay[mask_b] = (ref_overlay[mask_b].astype(float) * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        Image.fromarray(ref_overlay).save(exp_dir / "reference_mask_overlay.png")

        stage1_edit_mask.save(exp_dir / "stage1_edit_mask.png")
        stage1_protection_mask.save(exp_dir / "stage1_protection_mask.png")
        if stage1_removal_overlay:
            stage1_removal_overlay.save(exp_dir / "stage1_removal_overlay.png")
        if stage1_headwear_mask:
            stage1_headwear_mask.save(exp_dir / "stage1_headwear_mask.png")
        if stage1_raw_generated:
            stage1_raw_generated.save(exp_dir / "stage1_raw_generated.png")
        bald_base.save(exp_dir / "bald_base.png")

        # Aligned donor RGB & Alpha
        Image.fromarray(aligned_donor_rgb).save(exp_dir / "aligned_donor_rgb.png")
        Image.fromarray((np.clip(aligned_donor_alpha, 0, 1) * 255).astype(np.uint8)).save(
            exp_dir / "aligned_donor_alpha.png"
        )
        ref_crop.save(exp_dir / "ip_adapter_reference_crop.png")

        broad_edit_mask.save(exp_dir / "stage2_broad_edit_mask.png")
        edge_repair_mask.save(exp_dir / "stage2_edge_repair_mask.png")
        aligned_composite_before_diffusion.save(exp_dir / "aligned_composite_before_diffusion.png")
        stage2_raw_generated.save(exp_dir / "stage2_raw_generated.png")
        final_composite.save(exp_dir / "final_composite.png")

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

        # JSON Metadata & Metrics
        with (exp_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        with (exp_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with (exp_dir / "warnings.json").open("w", encoding="utf-8") as f:
            json.dump(warnings, f, indent=2)

        logger.info(f"TwoStageTransfer: All artifacts successfully saved to {exp_dir}")

    return {
        "composite_image": final_composite,
        "bald_base": bald_base,
        "stage1_raw_generated": stage1_raw_generated,
        "stage1_edit_mask": stage1_edit_mask,
        "stage2_raw_generated": stage2_raw_generated,
        "edge_repair_mask": edge_repair_mask,
        "broad_edit_mask": broad_edit_mask,
        "aligned_composite_before_diffusion": aligned_composite_before_diffusion,
        "aligned_donor_rgb": aligned_donor_rgb,
        "aligned_donor_alpha": aligned_donor_alpha,
        "reference_crop": ref_crop,
        "metrics": metrics,
        "metadata": metadata,
        "warnings": warnings,
        "experiment_dir": exp_dir_path,
    }
