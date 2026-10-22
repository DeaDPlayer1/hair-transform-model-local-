"""
Bald Converter Module for HairTryOn Two-Stage Architecture.

Removes the target's original hair and headwear, reconstructing a natural,
plausible bald/scalp base while strictly preserving face identity, features,
and unrelated background across arbitrary image resolutions and aspect ratios.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.headwear_detection import detect_headwear, create_removal_overlay, HeadwearDetectionResult

logger = logging.getLogger(__name__)

# Face parsing segmentation classes for strict protection:
# 2: l_eyebrow, 3: r_eyebrow, 4: l_eye, 5: r_eye, 6: glasses, 7: l_ear, 8: r_ear
# 10: nose, 11: mouth, 12: upper_lip, 13: lower_lip, 16: cloth
STRICT_PROTECTION_CLASSES = (2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16)
EYEBROW_CLASSES = (2, 3)
EYE_CLASSES = (4, 5, 6)
EAR_CLASSES = (7, 8)

# MediaPipe landmark indices for anatomical lower face contour:
JAWLINE_CONTOUR_INDICES = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
]
BROW_ARCH_INDICES = [
    454, 300, 293, 334, 296, 336, 168, 107, 66, 105, 63, 70, 234,
]


def compute_face_geometry(
    face_landmarks: Optional[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> Dict[str, Any]:
    """
    Extract head scale, pose, and lower face boundary from landmarks.
    Falls back gracefully if landmarks are missing or incomplete.
    """
    if not face_landmarks or not face_landmarks.get("detected"):
        fallback_scale = float(min(img_w, img_h) * 0.40)
        return {
            "detected": False,
            "face_scale": fallback_scale,
            "inter_ocular_dist": fallback_scale * 0.5,
            "face_width": fallback_scale,
            "face_height": fallback_scale * 1.2,
            "lower_face_poly": None,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
        }

    raw_pts = face_landmarks.get("landmarks_2d", [])
    if len(raw_pts) < 468:
        fallback_scale = float(min(img_w, img_h) * 0.40)
        return {
            "detected": True,
            "face_scale": fallback_scale,
            "inter_ocular_dist": fallback_scale * 0.5,
            "face_width": fallback_scale,
            "face_height": fallback_scale * 1.2,
            "lower_face_poly": None,
            "yaw": float(face_landmarks.get("yaw", 0.0)),
            "pitch": float(face_landmarks.get("pitch", 0.0)),
            "roll": float(face_landmarks.get("roll", 0.0)),
        }

    pts = np.array(raw_pts, dtype=np.float32)

    # Eye centers from eye corners
    r_eye = np.mean(pts[[33, 133]], axis=0)
    l_eye = np.mean(pts[[362, 263]], axis=0)
    iod = float(np.linalg.norm(r_eye - l_eye))

    # Face width (tragus to tragus)
    face_w = float(np.linalg.norm(pts[234] - pts[454]))

    # Face height (chin 152 to mid-forehead 10)
    face_h = float(np.linalg.norm(pts[152] - pts[10]))

    # Characteristic scale of the head/face in pixels
    face_scale = float(max(face_w, iod * 2.0))

    # Construct pose-adaptive lower face polygon
    valid_jaw = [idx for idx in JAWLINE_CONTOUR_INDICES if idx < len(pts)]
    valid_brow = [idx for idx in BROW_ARCH_INDICES if idx < len(pts)]
    if len(valid_jaw) >= 10 and len(valid_brow) >= 8:
        poly = np.vstack([pts[valid_jaw], pts[valid_brow]]).astype(np.int32)
    else:
        poly = None

    return {
        "detected": True,
        "face_scale": face_scale,
        "inter_ocular_dist": iod,
        "face_width": face_w,
        "face_height": face_h,
        "lower_face_poly": poly,
        "yaw": float(face_landmarks.get("yaw", 0.0)),
        "pitch": float(face_landmarks.get("pitch", 0.0)),
        "roll": float(face_landmarks.get("roll", 0.0)),
    }


def compute_inpaint_dimensions(
    w: int,
    h: int,
    max_dim: int = 768,
    min_dim: int = 512,
) -> Tuple[int, int]:
    """
    Compute aspect-ratio preserving dimensions divisible by 8 for Stable Diffusion UNet.
    Ensures latent inpainting runs without horizontal or vertical stretching/squashing.
    """
    aspect = w / h
    if aspect >= 1.0:
        new_w = min(max_dim, max(min_dim, w))
        new_h = int(round(new_w / aspect))
    else:
        new_h = min(max_dim, max(min_dim, h))
        new_w = int(round(new_h * aspect))

    # Clamp to boundaries
    new_w = min(max_dim, max(64, new_w))
    new_h = min(max_dim, max(64, new_h))

    # Ensure divisible by 8 (required for SD 8x VAE latent downsampling)
    new_w = max(64, (new_w // 8) * 8)
    new_h = max(64, (new_h // 8) * 8)
    return new_w, new_h


@dataclass
class BaldConverterConfig:
    """Configuration parameters for Stage 1 bald/scalp reconstruction."""

    backend: str = "auto"  # 'auto', 'sd-inpaint', or 'onnx'
    num_inference_steps: int = 12
    guidance_scale: float = 7.5
    hair_mask_expansion_ratio: float = 0.035  # fraction of face_scale
    max_hair_mask_expansion_px: int = 32
    feather_width_px: float = 3.0
    prompt: str = (
        "Photorealistic portrait of the same person, clean smooth bald head, shaved bald head, "
        "completely bald scalp, natural scalp skin, smooth skin on top of head, shaved bald hairstyle. "
        "Clean neutral wall background, natural lighting, hyperrealistic, high resolution."
    )
    negative_prompt: str = (
        "hair, stubble, bangs, fringe, hat, cap, baseball cap, beanie, helmet, headwear, "
        "wig, toupee, fuzzy hair, curly hair, long hair, hair strands, texture, distorted, blurry, "
        "deformed forehead, artifacts, colorful, purple"
    )
    onnx_model_path: Optional[str] = None
    headwear_mask_path: Optional[str] = None
    detect_headwear: bool = True
    seed: int = 42

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaldConverterConfig:
        """Create config from dictionary, ignoring unknown keys gracefully."""
        valid_keys = {
            "backend",
            "num_inference_steps",
            "guidance_scale",
            "hair_mask_expansion_ratio",
            "max_hair_mask_expansion_px",
            "feather_width_px",
            "prompt",
            "negative_prompt",
            "onnx_model_path",
            "headwear_mask_path",
            "detect_headwear",
            "seed",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> BaldConverterConfig:
        """Load configuration from JSON file."""
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        stage1_data = cfg.get("stage1", cfg)
        return cls.from_dict(stage1_data)


@dataclass
class BaldConversionResult:
    """Output container for Stage 1 bald reconstruction."""

    bald_base: Image.Image
    edit_mask: Image.Image
    protection_mask: Image.Image
    raw_generated: Optional[Image.Image] = None
    removal_overlay: Optional[Image.Image] = None
    headwear_mask: Optional[Image.Image] = None
    headwear_detected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def evaluate_stage1_quality_gate(
    original_target: np.ndarray,
    bald_base: np.ndarray,
    edit_mask: np.ndarray,
    protection_mask: np.ndarray,
    headwear_result: Optional[HeadwearDetectionResult] = None,
) -> Dict[str, Any]:
    """
    Programmatically verify Stage 1 outputs against quality standards.
    """
    h, w = original_target.shape[:2]
    total_pixels = h * w
    edit_binary = edit_mask > 128
    prot_binary = protection_mask > 128
    outside_binary = ~edit_binary

    # 1. Dimension match
    dim_match = (bald_base.shape[:2] == (h, w))

    # 2. Bit-exact outside preservation
    outside_count = int(np.count_nonzero(outside_binary))
    if outside_count > 0:
        diff_outside = np.abs(
            original_target[outside_binary, :3].astype(np.int32)
            - bald_base[outside_binary, :3].astype(np.int32)
        )
        exact_outside_count = int(np.count_nonzero(np.all(diff_outside == 0, axis=-1)))
        preservation_pct = (exact_outside_count / outside_count) * 100.0
        max_outside_diff = int(np.max(diff_outside))
    else:
        preservation_pct = 100.0
        max_outside_diff = 0

    # 3. Protection leak count
    prot_count = int(np.count_nonzero(prot_binary))
    if prot_count > 0:
        diff_prot = np.abs(
            original_target[prot_binary, :3].astype(np.int32)
            - bald_base[prot_binary, :3].astype(np.int32)
        )
        leak_count = int(np.count_nonzero(np.any(diff_prot > 0, axis=-1)))
    else:
        leak_count = 0

    # 4. Inpainted region plausibility (check non-blank, non-NaN/inf)
    edit_count = int(np.count_nonzero(edit_binary))
    edit_coverage_pct = (edit_count / total_pixels) * 100.0
    if edit_count > 0:
        inpainted_pixels = bald_base[edit_binary, :3]
        std_val = float(np.std(inpainted_pixels))
        has_nans = bool(np.isnan(inpainted_pixels).any() or np.isinf(inpainted_pixels).any())
    else:
        std_val = 0.0
        has_nans = False

    # 5. Quality status determination
    failure_reasons = []
    review_reasons = []

    if not dim_match:
        failure_reasons.append("Dimension mismatch between target and bald base")
    if preservation_pct < 99.99:
        failure_reasons.append(f"Outside pixel preservation compromised ({preservation_pct:.2f}% < 100%)")
    if leak_count > 0:
        review_reasons.append(f"Identity protection region has {leak_count} modified pixels")
    if has_nans:
        failure_reasons.append("NaN or Inf pixel values in bald base")
    if edit_count > 0 and std_val < 5.0:
        failure_reasons.append(f"Generated region has near-zero variance (std={std_val:.2f})")
    if edit_coverage_pct > 80.0:
        review_reasons.append(f"Implausibly large edit mask coverage ({edit_coverage_pct:.1f}%)")
    elif edit_coverage_pct < 1.0:
        review_reasons.append(f"Extremely small edit mask coverage ({edit_coverage_pct:.2f}%)")

    if failure_reasons:
        status = "FAILED"
    elif review_reasons:
        status = "NEEDS_VISUAL_REVIEW"
    else:
        status = "PASSED"

    return {
        "status": status,
        "preservation_pct": round(preservation_pct, 4),
        "max_outside_diff": max_outside_diff,
        "identity_leak_count": leak_count,
        "edit_coverage_pct": round(edit_coverage_pct, 2),
        "edit_pixels": edit_count,
        "headwear_detected": headwear_result.detected if headwear_result else False,
        "headwear_source": headwear_result.source if headwear_result else "none",
        "failure_reasons": failure_reasons,
        "review_reasons": review_reasons,
    }


class BaldConverter(ABC):
    """Abstract Base Class for bald conversion backends."""

    def __init__(self, config: BaldConverterConfig):
        self.config = config

    @abstractmethod
    def convert(
        self,
        target_image: Image.Image,
        target_hair_mask: Image.Image,
        target_segmentation_map: Optional[np.ndarray] = None,
        face_landmarks: Optional[Dict[str, Any]] = None,
        headwear_mask: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
        seed: Optional[int] = None,
    ) -> BaldConversionResult:
        """Perform hair and headwear removal and scalp generation."""
        pass

    def build_bald_edit_mask(
        self,
        target_image: Image.Image,
        target_hair_mask: Image.Image,
        target_segmentation_map: Optional[np.ndarray] = None,
        face_landmarks: Optional[Dict[str, Any]] = None,
        headwear_mask: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
        return_extended: bool = False,
    ) -> Union[
        Tuple[Image.Image, Image.Image, Image.Image, Any, List[str]],
        Tuple[Image.Image, Image.Image, List[str]],
    ]:
        """
        Dynamically construct Stage 1 edit and protection masks adapting to image
        dimensions, detected face scale, headwear, and facial pose.
        """
        warnings: List[str] = []
        w, h = target_image.size
        total_px = h * w

        # 1. Base hair mask binarization & dimension alignment
        hair_np = np.asarray(target_hair_mask.convert("L"))
        if hair_np.shape[:2] != (h, w):
            hair_np = cv2.resize(hair_np, (w, h), interpolation=cv2.INTER_NEAREST)
        hair_binary = (hair_np > 128).astype(np.uint8) * 255
        hair_px = int(np.count_nonzero(hair_binary))

        if hair_px == 0:
            warnings.append("Target hair mask contains 0 active pixels; target may already be bald.")
        elif (hair_px / total_px) > 0.85:
            warnings.append(f"Target hair mask covers {(hair_px/total_px)*100:.1f}% of image.")

        # 2. Extract anatomical face geometry & scale
        geom = compute_face_geometry(face_landmarks, w, h)
        face_scale = geom["face_scale"]

        # Scale-adaptive morphological margins (scaled to actual face size)
        expansion_px = int(np.clip(face_scale * self.config.hair_mask_expansion_ratio, 4, self.config.max_hair_mask_expansion_px))
        eyebrow_dilation_px = int(np.clip(face_scale * 0.015, 2, 8))
        close_px = int(np.clip(face_scale * 0.040, 5, 25))

        # 3. Headwear detection (Class 18 ONNX + manual mask + cranial check)
        manual_hw = headwear_mask if headwear_mask is not None else self.config.headwear_mask_path
        hw_result = detect_headwear(
            image=target_image,
            seg_map=target_segmentation_map,
            face_landmarks=face_landmarks,
            hair_mask=target_hair_mask,
            manual_mask=manual_hw,
            enable_heuristic=self.config.detect_headwear,
        )
        if hw_result.warnings:
            warnings.extend(hw_result.warnings)

        hw_np = np.asarray(hw_result.headwear_mask.convert("L"))
        if hw_np.shape[:2] != (h, w):
            hw_np = cv2.resize(hw_np, (w, h), interpolation=cv2.INTER_NEAREST)
        hw_binary = (hw_np > 128).astype(np.uint8) * 255

        if hw_result.detected:
            warnings.append(
                f"Headwear detected via '{hw_result.source}' covering {hw_result.pixel_count} px ({hw_result.coverage_pct:.2f}%)."
            )

        # Complete hair + headwear removal target
        combined_removal = np.maximum(hair_binary, hw_binary)

        # 4. Strict Facial Feature Protection
        features_mask = np.zeros((h, w), dtype=np.uint8)
        eyebrows_eyes_safe = np.zeros((h, w), dtype=np.uint8)

        if target_segmentation_map is not None:
            seg_map = target_segmentation_map
            if seg_map.shape[:2] != (h, w):
                seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)

            # Features: eyebrows, eyes, ears, nose, mouth, neck, clothing
            features_mask = np.isin(seg_map, STRICT_PROTECTION_CLASSES).astype(np.uint8) * 255

            # Dilate delicate eyebrows and eyes into skin only (never into hair/fringe)
            eyebrows_eyes = np.isin(seg_map, (2, 3, 4, 5, 6)).astype(np.uint8) * 255
            eb_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (eyebrow_dilation_px * 2 + 1, eyebrow_dilation_px * 2 + 1)
            )
            eyebrows_eyes_dilated = cv2.dilate(eyebrows_eyes, eb_kernel)
            eyebrows_eyes_safe = eyebrows_eyes_dilated & (~combined_removal)

        # 5. Pose-Adaptive Lower Face Protection
        # Landmark-driven lower face polygon: protects jawline, cheeks, nose, and brow arches,
        # leaving the upper forehead and fringe completely exposed for scalp generation.
        lower_face_mask = np.zeros((h, w), dtype=np.uint8)
        poly = geom.get("lower_face_poly")
        if poly is not None:
            cv2.fillPoly(lower_face_mask, [poly], 255)

        # Lower face skin is strictly protected, but hair/fringe is allowed to be removed
        lower_face_skin = lower_face_mask & (~combined_removal)

        # Strict identity protection combination
        strict_protection = np.maximum(features_mask, lower_face_skin)
        strict_protection = np.maximum(strict_protection, eyebrows_eyes_safe)

        # 6. Dynamic Editable Region Construction
        exp_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (expansion_px * 2 + 1, expansion_px * 2 + 1)
        )
        expanded_removal = cv2.dilate(combined_removal, exp_kernel)

        editable = expanded_removal.copy()
        editable[strict_protection > 0] = 0

        # Scale-adaptive hole closing inside hair/scalp dome (eliminates small gaps safely)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        editable = cv2.morphologyEx(editable, cv2.MORPH_CLOSE, close_kernel)

        # Re-enforce strict protection override
        editable[strict_protection > 0] = 0

        edit_mask_pil = Image.fromarray(editable, mode="L")
        protection_mask_pil = Image.fromarray(strict_protection, mode="L")
        removal_overlay_pil = create_removal_overlay(target_image, edit_mask_pil, color=(255, 30, 30), alpha=0.50)

        if return_extended:
            return edit_mask_pil, protection_mask_pil, removal_overlay_pil, hw_result, warnings
        return edit_mask_pil, protection_mask_pil, warnings


class SDInpaintBaldConverter(BaldConverter):
    """Stage 1 Bald Converter powered by Stable Diffusion 1.5 Inpainting."""

    def __init__(self, config: BaldConverterConfig, pipeline=None):
        super().__init__(config)
        self.pipeline = pipeline

    def _ensure_pipeline(self):
        if self.pipeline is None:
            logger.info("SDInpaintBaldConverter: Pipeline not provided. Loading local generative pipeline...")
            from src.local_generative_transfer import load_local_generative_pipeline

            self.pipeline = load_local_generative_pipeline()
        return self.pipeline

    def convert(
        self,
        target_image: Image.Image,
        target_hair_mask: Image.Image,
        target_segmentation_map: Optional[np.ndarray] = None,
        face_landmarks: Optional[Dict[str, Any]] = None,
        headwear_mask: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
        seed: Optional[int] = None,
    ) -> BaldConversionResult:
        t0 = time.perf_counter()
        w_orig, h_orig = target_image.size
        run_seed = seed if seed is not None else self.config.seed

        # Preserve alpha channel if present in input
        orig_alpha = None
        if target_image.mode == "RGBA":
            orig_alpha = target_image.split()[-1]
        target_rgb = target_image.convert("RGB")

        # Step 1: Construct dynamic edit and protection masks
        edit_mask_pil, protection_mask_pil, removal_overlay_pil, hw_result, warnings = (
            self.build_bald_edit_mask(
                target_image=target_rgb,
                target_hair_mask=target_hair_mask,
                target_segmentation_map=target_segmentation_map,
                face_landmarks=face_landmarks,
                headwear_mask=headwear_mask,
                return_extended=True,
            )
        )

        edit_np = np.asarray(edit_mask_pil)
        if np.count_nonzero(edit_np) == 0:
            warnings.append("Stage 1 edit mask is empty. Returning original target image as bald base.")
            return BaldConversionResult(
                bald_base=target_image.copy(),
                edit_mask=edit_mask_pil,
                protection_mask=protection_mask_pil,
                raw_generated=target_image.copy(),
                removal_overlay=removal_overlay_pil,
                headwear_mask=hw_result.headwear_mask,
                headwear_detected=hw_result.detected,
                metadata={
                    "backend": "sd-inpaint",
                    "status": "skipped_empty_mask",
                    "duration_seconds": time.perf_counter() - t0,
                },
                warnings=warnings,
            )

        pipe = self._ensure_pipeline()

        # Step 2: Compute aspect-ratio preserving dimensions divisible by 8
        inpaint_w, inpaint_h = compute_inpaint_dimensions(w_orig, h_orig, max_dim=768, min_dim=512)
        pipe_input_img = target_rgb.resize((inpaint_w, inpaint_h), Image.Resampling.LANCZOS)
        pipe_mask_img = edit_mask_pil.resize((inpaint_w, inpaint_h), Image.Resampling.NEAREST)

        import torch
        from diffusers import DDIMScheduler

        device = "cpu"
        if hasattr(pipe, "device") and isinstance(getattr(pipe, "device"), (str, torch.device)):
            device = pipe.device
        generator = torch.Generator(device=device).manual_seed(run_seed)

        # Temporarily unload IP-Adapter to prevent cross-attention artifacts during Stage 1 bald generation
        was_ip_adapter_loaded = False
        if hasattr(pipe, "unload_ip_adapter") and hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
            try:
                pipe.unload_ip_adapter()
                was_ip_adapter_loaded = True
            except Exception as e:
                logger.debug(f"Stage 1 IP-Adapter unload skipped: {e}")

        # Use DDIMScheduler for clean, artifact-free inpainting in Stage 1
        orig_scheduler = getattr(pipe, "scheduler", None)
        if (
            orig_scheduler is not None
            and hasattr(orig_scheduler, "config")
            and isinstance(orig_scheduler.config, dict)
        ):
            pipe.scheduler = DDIMScheduler.from_config(orig_scheduler.config)

        pipe_kwargs = {
            "prompt": self.config.prompt,
            "negative_prompt": self.config.negative_prompt,
            "image": pipe_input_img,
            "mask_image": pipe_mask_img,
            "num_inference_steps": max(10, self.config.num_inference_steps),
            "guidance_scale": self.config.guidance_scale,
            "generator": generator,
        }

        try:
            diffusion_output = pipe(**pipe_kwargs)
        finally:
            pipe.scheduler = orig_scheduler
            # Reload IP-Adapter for downstream Stage 2 usage
            if was_ip_adapter_loaded and hasattr(pipe, "load_ip_adapter"):
                try:
                    pipe.load_ip_adapter(
                        "h94/IP-Adapter",
                        subfolder="models",
                        weight_name="ip-adapter-plus_sd15.bin",
                        local_files_only=True,
                    )
                except Exception as e:
                    logger.debug(f"Stage 1 IP-Adapter reload deferred: {e}")

        raw_inpaint = diffusion_output.images[0]
        raw_generated = raw_inpaint.resize((w_orig, h_orig), Image.Resampling.LANCZOS)

        # Step 3: Validate generated image (check non-blank and finite)
        raw_np = np.asarray(raw_generated)
        if np.isnan(raw_np).any() or np.isinf(raw_np).any():
            raise ValueError("Stage 1 SD inpainting produced NaN or Inf pixel values.")
        if raw_np.std() < 1e-2:
            raise ValueError("Stage 1 SD inpainting produced a solid/blank image.")

        # Step 4: Strict masked compositing with scale-adaptive feather
        target_np = np.asarray(target_rgb)
        prot_np = np.asarray(protection_mask_pil)

        geom = compute_face_geometry(face_landmarks, w_orig, h_orig)
        face_scale = geom["face_scale"]
        feather = float(np.clip(face_scale * 0.010, 2.0, 10.0))

        dist = cv2.distanceTransform((edit_np > 128).astype(np.uint8), cv2.DIST_L2, 5)
        blend_alpha = np.clip(dist / feather, 0.0, 1.0).astype(np.float32)

        # Strictly zero out blend alpha outside edit mask and inside strict protection
        blend_alpha[edit_np == 0] = 0.0
        blend_alpha[prot_np > 0] = 0.0

        blend_alpha_3ch = blend_alpha[..., None]
        composite_np = (
            target_np.astype(np.float32) * (1.0 - blend_alpha_3ch)
            + raw_np.astype(np.float32) * blend_alpha_3ch
        ).clip(0, 255).astype(np.uint8)

        # Enforce exact bit preservation outside edit_mask
        outside_edit = edit_np == 0
        composite_np[outside_edit] = target_np[outside_edit]

        if orig_alpha is not None:
            # Reattach original alpha channel if target was RGBA
            alpha_np = np.asarray(orig_alpha)[..., None]
            composite_rgba = np.concatenate([composite_np, alpha_np], axis=-1)
            bald_base = Image.fromarray(composite_rgba, mode="RGBA")
        else:
            bald_base = Image.fromarray(composite_np, mode="RGB")

        duration = time.perf_counter() - t0

        # Step 5: Quality Gate Evaluation
        qg_metrics = evaluate_stage1_quality_gate(
            original_target=target_np,
            bald_base=composite_np,
            edit_mask=edit_np,
            protection_mask=prot_np,
            headwear_result=hw_result,
        )

        metadata = {
            "backend": "sd-inpaint",
            "seed": run_seed,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "duration_seconds": duration,
            "inpaint_dimensions": [inpaint_w, inpaint_h],
            "original_dimensions": [w_orig, h_orig],
            "face_scale": face_scale,
            "feather_px": feather,
            "quality_gate": qg_metrics,
            "headwear_detected": hw_result.detected,
            "headwear_source": hw_result.source,
            "headwear_pixels": hw_result.pixel_count,
        }

        if qg_metrics["status"] != "PASSED":
            warnings.append(f"Stage 1 Quality Gate: {qg_metrics['status']} ({qg_metrics.get('review_reasons', [])})")

        return BaldConversionResult(
            bald_base=bald_base,
            edit_mask=edit_mask_pil,
            protection_mask=protection_mask_pil,
            raw_generated=raw_generated,
            removal_overlay=removal_overlay_pil,
            headwear_mask=hw_result.headwear_mask,
            headwear_detected=hw_result.detected,
            metadata=metadata,
            warnings=warnings,
        )


class ONNXBaldConverter(BaldConverter):
    """
    Optional Stage 1 Bald Converter for genuine pretrained ONNX hair removal models.
    Validates model contracts and gracefully reports status.
    """

    def __init__(self, config: BaldConverterConfig):
        super().__init__(config)
        self.session = None
        self._init_session()

    def _init_session(self):
        model_path = self.config.onnx_model_path
        if not model_path:
            raise FileNotFoundError("ONNXBaldConverter: No onnx_model_path configured.")
        p = Path(model_path)
        if not p.is_file():
            raise FileNotFoundError(f"ONNXBaldConverter: Configured model file does not exist at '{p}'.")

        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            logger.info(f"ONNXBaldConverter: Loaded ONNX session from {p}")
        except Exception as e:
            raise RuntimeError(f"ONNXBaldConverter: Failed to load ONNX model: {e}") from e

    def convert(
        self,
        target_image: Image.Image,
        target_hair_mask: Image.Image,
        target_segmentation_map: Optional[np.ndarray] = None,
        face_landmarks: Optional[Dict[str, Any]] = None,
        headwear_mask: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
        seed: Optional[int] = None,
    ) -> BaldConversionResult:
        if self.session is None:
            raise RuntimeError("ONNXBaldConverter: Session is not initialized.")

        t0 = time.perf_counter()
        w_orig, h_orig = target_image.size
        edit_mask_pil, protection_mask_pil, removal_overlay_pil, hw_result, warnings = (
            self.build_bald_edit_mask(
                target_image=target_image,
                target_hair_mask=target_hair_mask,
                target_segmentation_map=target_segmentation_map,
                face_landmarks=face_landmarks,
                headwear_mask=headwear_mask,
                return_extended=True,
            )
        )

        input_meta = self.session.get_inputs()
        if len(input_meta) < 2:
            raise ValueError(
                f"ONNXBaldConverter: Expected at least 2 inputs (image, mask), found {len(input_meta)}"
            )

        img_512 = target_image.resize((512, 512), Image.Resampling.BILINEAR)
        img_arr = np.asarray(img_512, dtype=np.float32) / 255.0
        img_nchw = np.transpose(img_arr, (2, 0, 1))[None, ...]

        mask_512 = edit_mask_pil.resize((512, 512), Image.Resampling.NEAREST)
        mask_arr = (np.asarray(mask_512, dtype=np.float32) / 255.0)[None, None, ...]

        inputs = {input_meta[0].name: img_nchw, input_meta[1].name: mask_arr}
        outputs = self.session.run(None, inputs)
        raw_out = outputs[0]

        out_hwc = np.clip(np.transpose(raw_out[0], (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
        raw_generated = Image.fromarray(out_hwc).resize((w_orig, h_orig), Image.Resampling.LANCZOS)

        target_np = np.asarray(target_image.convert("RGB"))
        edit_np = np.asarray(edit_mask_pil)
        raw_np = np.asarray(raw_generated)

        composite_np = target_np.copy()
        composite_np[edit_np > 128] = raw_np[edit_np > 128]

        bald_base = Image.fromarray(composite_np, mode="RGB")
        duration = time.perf_counter() - t0

        return BaldConversionResult(
            bald_base=bald_base,
            edit_mask=edit_mask_pil,
            protection_mask=protection_mask_pil,
            raw_generated=raw_generated,
            removal_overlay=removal_overlay_pil,
            headwear_mask=hw_result.headwear_mask,
            headwear_detected=hw_result.detected,
            metadata={
                "backend": "onnx",
                "model_path": self.config.onnx_model_path,
                "duration_seconds": duration,
                "dimensions": [w_orig, h_orig],
            },
            warnings=warnings,
        )


def create_bald_converter(
    config: BaldConverterConfig,
    pipeline=None,
) -> BaldConverter:
    """Factory creating the appropriate BaldConverter implementation based on config."""
    backend = config.backend.lower().strip()
    if backend == "onnx":
        return ONNXBaldConverter(config)
    elif backend in ("sd-inpaint", "auto"):
        return SDInpaintBaldConverter(config, pipeline=pipeline)
    else:
        raise ValueError(
            f"Unsupported Stage 1 bald converter backend '{config.backend}'. Must be 'auto', 'sd-inpaint', or 'onnx'."
        )
