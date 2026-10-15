"""
Local Pretrained Diffusion Hairstyle Transfer Engine (CPU-Compatible).

Components:
1. Base Inpainting Pipeline: Stable Diffusion v1.5 Inpainting (runwayml/stable-diffusion-inpainting)
2. Conditioning Adapter: IP-Adapter Plus (h94/IP-Adapter with OpenCLIP ViT-H/14 image encoder)
3. High-Quality Hairstyle Preparation: Donor crop with neutral background
4. Target Edit Mask: 100% old hair coverage + cranial silhouette expansion + protected face
5. Feathered edge compositing preserving untouched target pixels (bit-exact 100% preservation)
6. Quantitative evaluation and experiment persistence (outputs/experiments/<experiment_id>/)
"""

import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

import cv2
import numpy as np
from PIL import Image
import psutil

# Ensure Hugging Face cache points to D:\HFCache if present
if "HF_HOME" not in os.environ and Path("D:/HFCache").is_dir():
    os.environ["HF_HOME"] = "D:/HFCache"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from transformers import CLIPVisionModelWithProjection
from diffusers import (
    StableDiffusionInpaintPipeline,
    DPMSolverMultistepScheduler,
)

from src.hair_mask_refinement import (
    build_refined_edit_mask,
    build_feathered_blend_mask,
    composite_with_feathering,
    refine_reference_hair_mask,
)
from src.transfer_evaluation import (
    evaluate_transfer_metrics,
    save_experiment_run,
    MANUAL_REVIEW_QUESTIONS,
)
from src.hairstyle_alignment import (
    align_reference_to_target,
    save_alignment_previews,
)


# Global singleton pipeline cache to avoid reloading weights repeatedly
_GLOBAL_PIPELINE: Optional[StableDiffusionInpaintPipeline] = None
_GLOBAL_PIPELINE_CONFIG: Dict[str, Any] = {}

# CPU Presets: calibrated for Intel i5-8350U CPU
INFERENCE_PRESETS = {
    "Fast Preview": {
        "steps": 8,
        "guidance_scale": 6.5,
        "description": "Fast ~60-90s CPU run. Good for quickly checking hairstyle placement and mask boundaries.",
    },
    "Balanced": {
        "steps": 10,
        "guidance_scale": 7.5,
        "description": "Balanced ~120-150s CPU run (Recommended). Crisp curl texture, natural blending, and solid identity preservation.",
    },
    "Quality": {
        "steps": 14,
        "guidance_scale": 7.5,
        "description": "High detail ~180-240s CPU run. Maximum hair strand fidelity and fine hairline definition.",
    },
}


def get_process_memory_mb() -> float:
    """Return the current resident set size (RSS) memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def prepare_reference_crop(
    reference_image: Image.Image,
    reference_hair_mask: Optional[Image.Image] = None,
    neutral_background: bool = True,
    context_padding: float = 0.15,
    output_size: Tuple[int, int] = (512, 512),
    legacy_gray_fade: bool = False,
) -> Image.Image:
    """
    Prepare a reference hairstyle crop emphasizing hair silhouette, volume, and curls.
    Decontaminates edge RGB values and uses a narrow, antialiased transition to neutral RGB 128
    so IP-Adapter conditions on crisp hair texture without a 4-pixel gray fade or background leakage.

    Args:
        reference_image: PIL Image of the hairstyle reference portrait.
        reference_hair_mask: Optional binary mask of reference hair (white=hair).
        neutral_background: If True, neutralizes background around hair to RGB (128, 128, 128).
        context_padding: Fractional padding around hair bounding box for head context.
        output_size: Output resolution tuple (512, 512) for IP-Adapter image encoder.
        legacy_gray_fade: If True, uses the previous distance/4.0 fade into 128 for baseline comparisons.
    """
    if not isinstance(reference_image, Image.Image):
        raise TypeError("reference_image must be a PIL Image instance.")

    ref_rgb = np.asarray(reference_image.convert("RGB"), dtype=np.uint8)
    h, w = ref_rgb.shape[:2]

    if reference_hair_mask is not None:
        mask_np = np.asarray(reference_hair_mask.convert("L"), dtype=np.uint8)
        if mask_np.shape[:2] != (h, w):
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_np, _ = refine_reference_hair_mask(ref_rgb, mask_np)
        hair_indices = mask_np > 128
    else:
        hair_indices = np.ones((h, w), dtype=bool)

    ys, xs = np.where(hair_indices)
    if len(xs) == 0:
        crop = ref_rgb
    else:
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bw = x_max - x_min + 1
        bh = y_max - y_min + 1

        pad_x = int(bw * context_padding)
        pad_y = int(bh * context_padding)

        crop_x1 = max(0, x_min - pad_x)
        crop_y1 = max(0, y_min - pad_y)
        crop_x2 = min(w, x_max + pad_x)
        crop_y2 = min(h, y_max + pad_y)

        crop = ref_rgb[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        if neutral_background and reference_hair_mask is not None:
            crop_mask = (mask_np[crop_y1:crop_y2, crop_x1:crop_x2] > 128).astype(np.uint8)
            neutral_canvas = np.full_like(crop, 128, dtype=np.uint8)

            if legacy_gray_fade:
                # Previous implementation: distance/4.0 ramp fading outer hair pixels into gray 128
                dist = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 5)
                alpha = np.clip(dist / 4.0, 0.0, 1.0)[..., None]
                crop = (crop.astype(np.float32) * alpha + neutral_canvas.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            else:
                # Targeted Fix:
                # 1. Edge Color Decontamination: remove donor background color bleed from boundary fringe
                k_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                core = cv2.erode(crop_mask, k_core, iterations=1)
                if np.any(core > 0):
                    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    dil = cv2.dilate(crop_mask, k_dil, iterations=1)
                    fringe_unknown = ((dil > 0) & (core == 0)).astype(np.uint8)
                    decontam_crop = cv2.inpaint(crop, fringe_unknown, 5, cv2.INPAINT_TELEA)
                else:
                    decontam_crop = crop.copy()

                # 2. Narrow Antialiased Boundary Transition:
                # signed_dist: positive inside hair mask, negative outside hair mask
                dist_in = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 5)
                dist_out = cv2.distanceTransform(1 - crop_mask, cv2.DIST_L2, 5)
                signed_dist = dist_in.astype(np.float32) - dist_out.astype(np.float32)
                # Narrow transition across +/- 0.6 pixels: full hair opacity down to boundary,
                # smooth antialiased seam, then clean neutral 128 canvas outside.
                alpha_narrow = np.clip((signed_dist + 0.6) / 1.2, 0.0, 1.0)[..., None]
                crop = (decontam_crop.astype(np.float32) * alpha_narrow + neutral_canvas.astype(np.float32) * (1.0 - alpha_narrow)).astype(np.uint8)

    crop_pil = Image.fromarray(crop, mode="RGB")
    return crop_pil.resize(output_size, Image.Resampling.LANCZOS)



def build_target_edit_mask(
    target_image: Image.Image,
    target_hair_mask: Image.Image,
    target_segmentation_map: Optional[np.ndarray] = None,
    face_landmarks: Optional[Dict[str, Any]] = None,
    cranial_expansion_px: int = 30,
) -> Image.Image:
    """
    Build an inpainting edit mask delegating to build_refined_edit_mask.
    Maintains backwards compatibility with existing callers and tests.
    """
    return build_refined_edit_mask(
        target_image=target_image,
        target_hair_mask=target_hair_mask,
        target_segmentation_map=target_segmentation_map,
        face_landmarks=face_landmarks,
        cranial_expansion_px=cranial_expansion_px,
        facial_feature_dilation_px=7,
        protect_clothing=True,
    )


def load_local_generative_pipeline(
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-inpainting",
    ip_adapter_repo: str = "h94/IP-Adapter",
    ip_adapter_weight: str = "ip-adapter-plus_sd15.bin",
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.float32,
    local_files_only: bool = True,
    force_reload: bool = False,
    progress_callback: Optional[callable] = None,
) -> StableDiffusionInpaintPipeline:
    """
    Load and cache the Stable Diffusion Inpainting pipeline with IP-Adapter Plus.
    Reuses cached model instances to prevent reloading overhead.
    Uses local_files_only=True by default to guarantee 100% offline local inference.
    """
    global _GLOBAL_PIPELINE, _GLOBAL_PIPELINE_CONFIG

    current_config = {
        "model_id": model_id,
        "ip_adapter_repo": ip_adapter_repo,
        "ip_adapter_weight": ip_adapter_weight,
        "device": device,
    }

    if (
        not force_reload
        and _GLOBAL_PIPELINE is not None
        and _GLOBAL_PIPELINE_CONFIG == current_config
    ):
        return _GLOBAL_PIPELINE

    if progress_callback:
        progress_callback("Loading OpenCLIP ViT-H/14 image encoder for IP-Adapter Plus...")

    # 1. Load CLIP Image Encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        ip_adapter_repo,
        subfolder="models/image_encoder",
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )

    if progress_callback:
        progress_callback("Loading Stable Diffusion v1.5 Inpainting pipeline (CPU float32)...")

    # 2. Load Base Inpainting Pipeline
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        image_encoder=image_encoder,
        torch_dtype=torch_dtype,
        safety_checker=None,
        local_files_only=local_files_only,
    )

    # 3. High-Efficiency DPMSolverMultistepScheduler for 8-14 step inference
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        use_karras_sigmas=True,
    )

    if progress_callback:
        progress_callback("Loading IP-Adapter Plus cross-attention weights...")

    # 4. Load IP-Adapter Plus weights
    pipeline.load_ip_adapter(
        ip_adapter_repo,
        subfolder="models",
        weight_name=ip_adapter_weight,
        local_files_only=local_files_only,
    )

    # 5. Send to device
    # CRITICAL: DO NOT call enable_attention_slicing() here as it causes
    # IP-Adapter's attention processor to fail with AttributeError on tuples.
    pipeline.to(device)

    _GLOBAL_PIPELINE = pipeline
    _GLOBAL_PIPELINE_CONFIG = current_config
    return pipeline


def generate_local_hairstyle_transfer(
    target_image: Image.Image,
    reference_image: Image.Image,
    target_hair_mask: Image.Image,
    reference_hair_mask: Optional[Image.Image] = None,
    target_segmentation_map: Optional[np.ndarray] = None,
    face_landmarks: Optional[Dict[str, Any]] = None,
    reference_landmarks: Optional[Dict[str, Any]] = None,
    user_instruction: Optional[str] = None,
    negative_prompt_extra: Optional[str] = None,
    num_inference_steps: int = 10,
    ip_adapter_scale: float = 0.75,
    guidance_scale: float = 7.5,
    seed: int = 42,
    cranial_expansion_px: int = 30,
    feather_width_px: float = 6.0,
    hairline_transition_px: int = 15,
    neutral_reference_bg: bool = True,
    use_alignment: bool = True,
    alignment_only: bool = False,
    save_experiment: bool = True,
    experiment_id: Optional[str] = None,
    pipeline: Optional[StableDiffusionInpaintPipeline] = None,
    progress_callback: Optional[callable] = None,
    legacy_crop: bool = False,
    bounded_decontamination: bool = True,
) -> Dict[str, Any]:
    """
    Run local generative hairstyle transfer using Stable Diffusion Inpainting + IP-Adapter Plus.

    Args:
        target_image: Portrait of the target person whose identity/face must be preserved.
        reference_image: Reference photo showing the desired hairstyle.
        target_hair_mask: Binary mask of target hair (white=hair).
        reference_hair_mask: Optional binary mask of reference hair.
        target_segmentation_map: Full 19-class parsing map for facial feature protection.
        face_landmarks: MediaPipe facial landmarks dict.
        user_instruction: Custom hairstyle description or tweaks.
        negative_prompt_extra: Additional negative prompt keywords.
        num_inference_steps: Number of diffusion inpainting steps.
        ip_adapter_scale: Conditioning scale (0.0 to 1.0).
        guidance_scale: Classifier-free guidance scale.
        seed: Random seed for reproducible generation.
        cranial_expansion_px: Pixels to expand edit mask upwards and sideways.
        feather_width_px: Width in pixels for smooth inner feathering.
        neutral_reference_bg: If True, neutralizes background of reference crop.
        save_experiment: If True, records run to outputs/experiments/<experiment_id>/.
        experiment_id: Optional custom identifier.
        pipeline: Pre-loaded pipeline instance.
        progress_callback: Callback for status updates.
        legacy_crop: If True, uses the previous distance/4.0 gray fade for reference crop.
        bounded_decontamination: If True, bounds OpenCV inpainting to a narrow edge ring.

    Returns:
        dict containing:
            'composite_image': Final blended PIL image with feathered edge
            'raw_generated': Raw diffusion output image (full target dimensions)
            'edit_mask': Binary edit mask used for inpainting
            'reference_crop': Prepared reference hairstyle conditioning image
            'blend_alpha': Feathered alpha array
            'metrics': Quantitative evaluation metrics
            'metadata': Complete reproducible run parameters
            'experiment_dir': Path to saved experiment directory (if saved)
            'duration_seconds': Total inference time
            'peak_memory_mb': Memory usage
            'seed': Random seed used
            'steps': Number of inference steps
            'ip_adapter_scale': IP-Adapter scale
    """
    if target_image is None or reference_image is None or target_hair_mask is None:
        raise ValueError("target_image, reference_image, and target_hair_mask are all required.")

    start_time = time.perf_counter()
    w_orig, h_orig = target_image.size

    # 1. Prepare Reference Hairstyle Conditioning Image
    if progress_callback:
        progress_callback("Preparing reference hairstyle crop and conditioning...")
    ref_crop = prepare_reference_crop(
        reference_image=reference_image,
        reference_hair_mask=reference_hair_mask,
        neutral_background=neutral_reference_bg,
        context_padding=0.15,
        output_size=(512, 512),
        legacy_gray_fade=legacy_crop,
    )

    # 2. Geometry-Aware Alignment (spatial prior)
    alignment_result = None
    if use_alignment and reference_hair_mask is not None:
        if progress_callback:
            progress_callback("Aligning reference hairstyle to target head geometry...")

        base_dir = Path(__file__).resolve().parent.parent
        preview_dir = str(base_dir / "outputs" / "alignment_previews")

        alignment_result = align_reference_to_target(
            target_image=target_image,
            reference_image=reference_image,
            target_hair_mask=target_hair_mask,
            reference_hair_mask=reference_hair_mask,
            target_landmarks=face_landmarks,
            reference_landmarks=reference_landmarks,
            target_segmentation_map=target_segmentation_map,
            cranial_expansion_px=cranial_expansion_px,
            feather_width_px=feather_width_px,
            hairline_transition_px=hairline_transition_px,
            save_previews_dir=preview_dir,
            bounded_decontamination=bounded_decontamination,
        )

        # Base portrait on which inpainting operates: has the aligned reference curls pre-composited
        inpainting_base = alignment_result.get("pasted_composite", target_image)
        # Dedicated narrow edge-repair mask around hair boundary + uncovered old hair
        edit_mask = alignment_result.get("edge_repair_mask", alignment_result["target_edit_mask"])
        broad_edit_mask = alignment_result.get("broad_edit_mask", alignment_result["target_edit_mask"])

        if alignment_result.get("warnings"):
            for w in alignment_result["warnings"]:
                print(f"Alignment warning: {w}")
    else:
        # Fallback: use the original mask-only edit mask
        if progress_callback:
            progress_callback("Building target edit mask with facial protection...")
        edit_mask = build_target_edit_mask(
            target_image=target_image,
            target_hair_mask=target_hair_mask,
            target_segmentation_map=target_segmentation_map,
            face_landmarks=face_landmarks,
            cranial_expansion_px=cranial_expansion_px,
        )
        inpainting_base = target_image
        broad_edit_mask = edit_mask
        alignment_result = None

    # Validate mask is not empty and not full image
    mask_arr = np.asarray(edit_mask)
    edit_pixels = np.count_nonzero(mask_arr > 128)
    if edit_pixels == 0:
        raise ValueError("Edit mask is empty. Hair segmentation did not detect any editable regions.")
    if edit_pixels >= (w_orig * h_orig * 0.95):
        raise ValueError("Edit mask covers nearly the entire image (>95%). Refusing to run inpainting without facial protection.")

    # If alignment_only mode, return previews without running diffusion
    if alignment_only:
        composite_to_return = inpainting_base if alignment_result is not None else target_image
        return {
            "composite_image": composite_to_return,
            "raw_generated": composite_to_return,
            "edit_mask": broad_edit_mask,
            "edge_repair_mask": edit_mask,
            "reference_crop": ref_crop,
            "blend_alpha": None,
            "metrics": {},
            "metadata": {"alignment_only": True},
            "experiment_dir": None,
            "duration_seconds": time.perf_counter() - start_time,
            "peak_memory_mb": get_process_memory_mb(),
            "seed": seed,
            "steps": 0,
            "ip_adapter_scale": ip_adapter_scale,
            "alignment_result": alignment_result,
            "manual_review_questions": [],
        }

    # 3. Resize Inpainting Base & Narrow Edge Mask to 512x512 for SD Inpainting
    base_512 = inpainting_base.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    mask_512 = edit_mask.resize((512, 512), Image.Resampling.NEAREST)

    # Save exact inpainting model inputs and intermediate validation artifacts
    base_dir = Path(__file__).resolve().parent.parent
    preview_dir = base_dir / "outputs" / "alignment_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    base_512.save(preview_dir / "diffusion_input_image.png")
    mask_512.save(preview_dir / "diffusion_input_mask.png")
    ref_crop.save(preview_dir / "ip_adapter_ref_crop.png")
    if alignment_result and "pasted_composite" in alignment_result:
        alignment_result["pasted_composite"].save(preview_dir / "pasted_composite.png")
    edit_mask.save(preview_dir / "edge_repair_mask.png")

    # 4. Load or use provided Pipeline
    if pipeline is None:
        if progress_callback:
            progress_callback("Initializing local diffusion pipeline...")
        pipe = load_local_generative_pipeline(progress_callback=progress_callback)
    else:
        pipe = pipeline

    # 5. Set IP-Adapter conditioning scale
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    # 6. Prompts
    prompt_base = (
        "photograph of a person with the new hairstyle, high quality portrait, "
        "natural hairline, photorealistic, 8k uhd, sharp focus, detailed hair strands, "
        "seamless scalp transition"
    )
    if user_instruction and user_instruction.strip():
        prompt = f"{prompt_base}, {user_instruction.strip()}"
    else:
        prompt = prompt_base

    negative_prompt_base = (
        "old curls underneath, duplicated hairline, double hair, wig, toupee, cap, "
        "bald patches, white gaps, distorted face, deformed eyes, extra ears, blurry, "
        "low quality, painting, cartoon, artifacts, bad anatomy"
    )
    if negative_prompt_extra and negative_prompt_extra.strip():
        negative_prompt = f"{negative_prompt_base}, {negative_prompt_extra.strip()}"
    else:
        negative_prompt = negative_prompt_base

    device_to_use = getattr(pipe, "device", torch.device("cpu"))
    generator = torch.Generator(device=device_to_use).manual_seed(seed)

    if progress_callback:
        progress_callback(f"Running CPU diffusion inpainting ({num_inference_steps} steps, scale={ip_adapter_scale})...")

    # 7. Run Denoising Inpainting
    with torch.inference_mode():
        diffusion_output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_512,
            mask_image=mask_512,
            ip_adapter_image=ref_crop,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    raw_generated_512 = diffusion_output.images[0]
    raw_generated_orig = raw_generated_512.resize((w_orig, h_orig), Image.Resampling.LANCZOS)

    # 8. Feathered Boundary Compositing
    # Blends diffusion inpainting strictly into the narrow edge repair seam
    # while preserving the central reference hair pixels and curls 100% intact!
    final_composite, blend_alpha = composite_with_feathering(
        target_image=inpainting_base,
        generated_image=raw_generated_orig,
        edit_mask=edit_mask,
        feather_width_px=feather_width_px,
    )

    # Strictly guarantee that pixels outside the broad hair envelope
    # are 100% bit-exact preserved from the original target portrait
    if broad_edit_mask is not None:
        broad_mask_arr = np.asarray(broad_edit_mask.convert("L"))
        outside_broad = broad_mask_arr == 0
        final_arr = np.asarray(final_composite).copy()
        target_arr = np.asarray(target_image).copy()
        final_arr[outside_broad] = target_arr[outside_broad]
        final_composite = Image.fromarray(final_arr)

    duration = time.perf_counter() - start_time
    peak_mem = get_process_memory_mb()

    # 9. Quantitative Evaluation Metrics
    metrics = evaluate_transfer_metrics(
        target_image=target_image,
        composite_image=final_composite,
        edit_mask=broad_edit_mask if broad_edit_mask is not None else edit_mask,
        target_segmentation_map=target_segmentation_map,
        duration_seconds=duration,
        peak_memory_mb=peak_mem,
    )

    # 10. Record Metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "ip_adapter_weight": "ip-adapter-plus_sd15.bin",
        "device": str(getattr(pipe, "device", "cpu")),
        "target_size": [w_orig, h_orig],
        "inference_resolution": [512, 512],
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "ip_adapter_scale": ip_adapter_scale,
        "seed": seed,
        "cranial_expansion_px": cranial_expansion_px,
        "feather_width_px": feather_width_px,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "duration_seconds": round(duration, 2),
        "peak_memory_mb": round(peak_mem, 1),
    }

    # 11. Optionally persist experiment run
    exp_dir = None
    if save_experiment:
        base_dir = Path(__file__).resolve().parent.parent
        outputs_dir = base_dir / "outputs"
        try:
            exp_dir = save_experiment_run(
                outputs_base_dir=outputs_dir,
                target_image=target_image,
                reference_image=reference_image,
                hair_mask=target_hair_mask,
                edit_mask=edit_mask,
                composite_image=final_composite,
                metadata=metadata,
                metrics=metrics,
                experiment_id=experiment_id,
            )
        except Exception as exc:
            print(f"Warning: Failed to save experiment to disk: {exc}")

    return {
        "composite_image": final_composite,
        "raw_generated": raw_generated_orig,
        "edit_mask": broad_edit_mask,
        "edge_repair_mask": edit_mask,
        "reference_crop": ref_crop,
        "blend_alpha": blend_alpha,
        "metrics": metrics,
        "metadata": metadata,
        "experiment_dir": exp_dir,
        "duration_seconds": duration,
        "peak_memory_mb": peak_mem,
        "seed": seed,
        "steps": num_inference_steps,
        "ip_adapter_scale": ip_adapter_scale,
        "alignment_result": alignment_result,
        "manual_review_questions": MANUAL_REVIEW_QUESTIONS,
    }
