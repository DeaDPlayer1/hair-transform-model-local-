"""
Bald Converter Module for HairTryOn Two-Stage Architecture.

Removes the target's original hair and reconstructs a natural, plausible
bald/scalp base while strictly preserving face identity, features, and background.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.headwear_detection import detect_headwear, create_removal_overlay

logger = logging.getLogger(__name__)

# Face parsing segmentation classes for strict protection:
# 2: l_eyebrow, 3: r_eyebrow, 4: l_eye, 5: r_eye, 6: glasses, 7: l_ear, 8: r_ear
# 10: nose, 11: mouth, 12: upper_lip, 13: lower_lip, 16: cloth
STRICT_PROTECTION_CLASSES = (2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16)
EAR_CLASSES = (7, 8)


@dataclass
class BaldConverterConfig:
    """Configuration parameters for Stage 1 bald/scalp reconstruction."""

    backend: str = "auto"  # 'auto', 'sd-inpaint', or 'onnx'
    num_inference_steps: int = 12
    guidance_scale: float = 7.5
    hair_mask_expansion_ratio: float = 0.008
    max_hair_mask_expansion_px: int = 10
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
        Construct the Stage 1 edit mask with scale-aware margin, headwear detection,
        and strict facial feature protection without carving out forehead bangs/fringe.

        Returns:
            If return_extended=True:
                edit_mask_pil, protection_mask_pil, removal_overlay_pil, hw_result, warnings
            If return_extended=False:
                edit_mask_pil, protection_mask_pil, warnings
        """
        warnings: List[str] = []
        w, h = target_image.size

        # 1. Base hair mask binarization
        hair_np = np.asarray(target_hair_mask.convert("L"))
        if hair_np.shape[:2] != (h, w):
            hair_np = cv2.resize(hair_np, (w, h), interpolation=cv2.INTER_NEAREST)
        hair_binary = (hair_np > 128).astype(np.uint8) * 255
        hair_px = int(np.count_nonzero(hair_binary))
        total_px = h * w

        if hair_px == 0:
            warnings.append("Target hair mask contains 0 active pixels; target may already be bald.")
        elif (hair_px / total_px) > 0.85:
            warnings.append(f"Target hair mask covers {(hair_px/total_px)*100:.1f}% of image.")

        # 2. Headwear detection (Class 18 ONNX + compatible heuristic + manual mask)
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

        # Combine hair + headwear into complete removal target
        combined_removal = np.maximum(hair_binary, hw_binary)

        # 3. Scale-aware expansion margin (ensures outer flyaways and silhouette edges are covered)
        expansion_px = max(
            4,
            min(
                self.config.max_hair_mask_expansion_px,
                int(round(max(w, h) * self.config.hair_mask_expansion_ratio)),
            ),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion_px * 2 + 1, expansion_px * 2 + 1))
        expanded_removal = cv2.dilate(combined_removal, kernel, iterations=1)

        # 4. Strict Facial Feature Protection (eyebrows, eyes, nose, mouth, ears, clothing)
        # Eyebrows (classes 2, 3), Eyes (4, 5, 6), Ears (7, 8), Nose (10), Lips/Mouth (11, 12, 13), Cloth (16)
        vital_protection = np.zeros((h, w), dtype=np.uint8)

        if target_segmentation_map is not None:
            seg_map = target_segmentation_map
            if seg_map.shape[:2] != (h, w):
                seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)

            # Features: eyebrows, eyes, ears, nose, mouth, clothing
            features_mask = np.isin(seg_map, STRICT_PROTECTION_CLASSES).astype(np.uint8) * 255
            # Dilate delicate eyebrows and eyes into skin only (never into hair/fringe)
            eyebrows_eyes = np.isin(seg_map, (2, 3, 4, 5, 6)).astype(np.uint8) * 255
            eyebrows_eyes_dilated = cv2.dilate(eyebrows_eyes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            eyebrows_eyes_safe = eyebrows_eyes_dilated & (~combined_removal)
            vital_protection = np.maximum(features_mask, eyebrows_eyes_safe)

        # 5. Lower face hull protection (protect cheeks, chin, jawline below eyebrow level)
        # CRITICAL: We do NOT include the upper forehead in this protection hull, because
        # doing so would carve out forehead bangs/fringe and leave them behind!
        lower_face_mask = np.zeros((h, w), dtype=np.uint8)
        if face_landmarks and face_landmarks.get("detected"):
            landmarks_2d = face_landmarks.get("landmarks_2d", [])
            if len(landmarks_2d) >= 10:
                pts = np.array(landmarks_2d, dtype=np.int32)
                # Eyebrow landmark points to establish horizontal divide
                eyebrow_indices = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
                valid_idx = [i for i in eyebrow_indices if i < len(pts)]
                if valid_idx:
                    eyebrow_pts = pts[valid_idx]
                    min_brow_y = int(np.min(eyebrow_pts[:, 1]))
                    # Lower face: landmarks at or below eyebrow level minus 4px safety buffer
                    lower_pts = pts[pts[:, 1] >= (min_brow_y - 4)]
                    if len(lower_pts) >= 6:
                        lower_hull = cv2.convexHull(lower_pts)
                        cv2.fillConvexPoly(lower_face_mask, lower_hull, 255)

        # Lower face skin is protected, but hair/fringe/cap itself is allowed to be edited
        # unless it directly touches vital features (eyebrows/eyes/lips/ears/cloth)
        protection = np.maximum(vital_protection, lower_face_mask & (~combined_removal))

        # 6. Construct Editable Region: expanded removal minus strict protection
        editable = expanded_removal.copy()
        editable[vital_protection > 0] = 0

        # Fill small holes inside editable hair/headwear region for contiguous scalp synthesis
        if np.any(editable > 0):
            flood = editable.copy()
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flood, flood_mask, (0, 0), 255)
            inv_flood = cv2.bitwise_not(flood)
            editable = (editable | inv_flood)
            editable[vital_protection > 0] = 0

        edit_mask_pil = Image.fromarray(editable, mode="L")
        protection_mask_pil = Image.fromarray(protection, mode="L")
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

        # Step 1: Construct edit and protection masks with headwear and fringe coverage
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

        # Step 2: Prepare inputs for 512x512 diffusion inpainting
        pipe_input_img = target_image.resize((512, 512), Image.Resampling.LANCZOS)
        pipe_mask_img = edit_mask_pil.resize((512, 512), Image.Resampling.NEAREST)

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
            # Restore scheduler
            pipe.scheduler = orig_scheduler
            # If IP-Adapter was unloaded, reload it for downstream Stage 2 usage
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

        raw_512 = diffusion_output.images[0]
        raw_generated = raw_512.resize((w_orig, h_orig), Image.Resampling.LANCZOS)

        # Step 3: Validate generated image (check non-blank and finite)
        raw_np = np.asarray(raw_generated)
        if np.isnan(raw_np).any() or np.isinf(raw_np).any():
            raise ValueError("Stage 1 SD inpainting produced NaN or Inf pixel values.")
        if raw_np.std() < 1e-2:
            raise ValueError("Stage 1 SD inpainting produced a solid/blank image.")

        # Step 4: Strict masked compositing and preservation
        target_np = np.asarray(target_image.convert("RGB"))
        prot_np = np.asarray(protection_mask_pil)

        # Smooth feather transition inside editable region only
        feather = max(1.0, float(self.config.feather_width_px))
        dist = cv2.distanceTransform((edit_np > 128).astype(np.uint8), cv2.DIST_L2, 5)
        blend_alpha = np.clip(dist / feather, 0.0, 1.0).astype(np.float32)

        # Strictly zero out blend alpha where edit_mask is 0 or protection is > 0
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

        bald_base = Image.fromarray(composite_np, mode="RGB")
        duration = time.perf_counter() - t0

        metadata = {
            "backend": "sd-inpaint",
            "seed": run_seed,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "duration_seconds": duration,
            "edit_mask_coverage_pct": (np.count_nonzero(edit_np > 128) / (w_orig * h_orig)) * 100.0,
            "headwear_detected": hw_result.detected,
            "headwear_source": hw_result.source,
            "headwear_pixels": hw_result.pixel_count,
            "dimensions": [w_orig, h_orig],
        }

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
        seed: Optional[int] = None,
    ) -> BaldConversionResult:
        if self.session is None:
            raise RuntimeError("ONNXBaldConverter: Session is not initialized.")

        t0 = time.perf_counter()
        w_orig, h_orig = target_image.size
        edit_mask_pil, protection_mask_pil, warnings = self.build_bald_edit_mask(
            target_image=target_image,
            target_hair_mask=target_hair_mask,
            target_segmentation_map=target_segmentation_map,
            face_landmarks=face_landmarks,
            return_extended=False,
        )

        # Contract validation on inputs
        input_meta = self.session.get_inputs()
        if len(input_meta) < 2:
            raise ValueError(
                f"ONNXBaldConverter: Expected at least 2 inputs (image, mask), found {len(input_meta)}"
            )

        # Standard ONNX preprocessing (512x512 NCHW float32 [0, 1])
        img_512 = target_image.resize((512, 512), Image.Resampling.BILINEAR)
        img_arr = np.asarray(img_512, dtype=np.float32) / 255.0
        img_nchw = np.transpose(img_arr, (2, 0, 1))[None, ...]

        mask_512 = edit_mask_pil.resize((512, 512), Image.Resampling.NEAREST)
        mask_arr = (np.asarray(mask_512, dtype=np.float32) / 255.0)[None, None, ...]

        inputs = {input_meta[0].name: img_nchw, input_meta[1].name: mask_arr}
        outputs = self.session.run(None, inputs)
        raw_out = outputs[0]

        # Postprocessing NCHW -> HWC uint8
        out_hwc = np.clip(np.transpose(raw_out[0], (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
        raw_generated = Image.fromarray(out_hwc).resize((w_orig, h_orig), Image.Resampling.LANCZOS)

        # Strict masked composite
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
            metadata={
                "backend": "onnx",
                "model_path": self.config.onnx_model_path,
                "duration_seconds": duration,
                "dimensions": [w_orig, h_orig],
            },
            warnings=warnings,
        )


def create_bald_converter(
    config: Optional[Union[BaldConverterConfig, Dict[str, Any]]] = None,
    pipeline=None,
) -> BaldConverter:
    """
    Factory function for Stage 1 Bald Converter.

    Resolves 'auto' backend:
    - If a valid ONNX bald model is configured and available, uses ONNXBaldConverter.
    - Otherwise, cleanly logs and uses SDInpaintBaldConverter.
    """
    if config is None:
        cfg = BaldConverterConfig()
    elif isinstance(config, dict):
        cfg = BaldConverterConfig.from_dict(config)
    else:
        cfg = config

    backend = cfg.backend.lower().strip()

    if backend == "onnx":
        logger.info(f"Explicit ONNX backend requested: {cfg.onnx_model_path}")
        return ONNXBaldConverter(cfg)

    if backend == "sd-inpaint":
        logger.info("Using Stable Diffusion Inpainting bald converter backend.")
        return SDInpaintBaldConverter(cfg, pipeline=pipeline)

    # backend == 'auto': Check if ONNX model is available
    if cfg.onnx_model_path and Path(cfg.onnx_model_path).is_file():
        try:
            logger.info(f"Auto-selected ONNX backend from valid model path: {cfg.onnx_model_path}")
            return ONNXBaldConverter(cfg)
        except Exception as e:
            logger.warning(f"Failed to initialize ONNX model ({e}). Falling back to SD Inpainting.")

    logger.info(
        "Stage 1 Bald Converter: Auto-selected SD Inpainting backend "
        "(no compatible pretrained ONNX bald model found; using Stable Diffusion 1.5 inpainting fallback)."
    )
    return SDInpaintBaldConverter(cfg, pipeline=pipeline)
