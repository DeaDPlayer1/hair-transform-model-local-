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

logger = logging.getLogger(__name__)

# Face parsing segmentation classes for strict protection
# 2: skin, 3: l_eyebrow, 4: r_eyebrow, 5: l_eye, 6: r_eye, 7: l_ear, 8: r_ear
# 10: nose, 11: upper_lip, 12: lower_lip, 13: mouth, 16: cloth
STRICT_PROTECTION_CLASSES = (2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16)
EAR_CLASSES = (7, 8)


@dataclass
class BaldConverterConfig:
    """Configuration parameters for Stage 1 bald/scalp reconstruction."""

    backend: str = "auto"  # 'auto', 'sd-inpaint', or 'onnx'
    num_inference_steps: int = 8
    guidance_scale: float = 7.5
    hair_mask_expansion_ratio: float = 0.01
    max_hair_mask_expansion_px: int = 12
    feather_width_px: float = 2.0
    prompt: str = (
        "Photorealistic portrait of the same person with naturally bald, clean scalp skin. "
        "Preserve the exact facial identity, forehead proportions, eyebrows, eyes, ears, "
        "expression, lighting, camera angle, clothing and background."
    )
    negative_prompt: str = (
        "hair, loose hair strands, wig, hair cap, distorted scalp, changed face, "
        "changed eyebrows, changed eyes, altered ears, deformed forehead, blurry skin, "
        "artificial skin, artifacts"
    )
    onnx_model_path: Optional[str] = None
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
        seed: Optional[int] = None,
    ) -> BaldConversionResult:
        """Perform hair removal and scalp generation."""
        pass

    def build_bald_edit_mask(
        self,
        target_image: Image.Image,
        target_hair_mask: Image.Image,
        target_segmentation_map: Optional[np.ndarray] = None,
        face_landmarks: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Image.Image, Image.Image, List[str]]:
        """
        Construct the Stage 1 edit mask with scale-aware margin and strict feature protection.

        Returns:
            edit_mask_pil: Editable region (255 = inpaint scalp, 0 = protected)
            protection_mask_pil: Strict facial/feature protection mask (255 = protected)
            warnings: Diagnostic warnings if masks required adjustment
        """
        warnings: List[str] = []
        w, h = target_image.size
        hair_np = np.asarray(target_hair_mask.convert("L"))
        if hair_np.shape[:2] != (h, w):
            hair_np = cv2.resize(hair_np, (w, h), interpolation=cv2.INTER_NEAREST)

        hair_binary = (hair_np > 128).astype(np.uint8) * 255
        hair_px = int(np.count_nonzero(hair_binary))
        total_px = h * w

        if hair_px == 0:
            warnings.append("Target hair mask contains 0 active pixels; target may already be bald.")
        elif (hair_px / total_px) > 0.85:
            warnings.append(f"Target hair mask covers {(hair_px/total_px)*100:.1f}% of image, which is unusually large.")

        # 1. Scale-aware expansion margin for hair boundary
        expansion_px = max(
            1,
            min(
                self.config.max_hair_mask_expansion_px,
                int(round(max(w, h) * self.config.hair_mask_expansion_ratio)),
            ),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion_px * 2 + 1, expansion_px * 2 + 1))
        expanded_hair = cv2.dilate(hair_binary, kernel, iterations=1)

        # 2. Strict Protection Mask
        protection = np.zeros((h, w), dtype=np.uint8)

        # Segmented facial features (ears, eyes, brows, nose, mouth, clothing)
        if target_segmentation_map is not None:
            seg_map = target_segmentation_map
            if seg_map.shape[:2] != (h, w):
                seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)
            features_mask = np.isin(seg_map, STRICT_PROTECTION_CLASSES).astype(np.uint8) * 255
            protection = np.maximum(protection, features_mask)

        # Landmark facial hull protection
        if face_landmarks and face_landmarks.get("detected"):
            landmarks_2d = face_landmarks.get("landmarks_2d", [])
            if len(landmarks_2d) >= 10:
                pts = np.array(landmarks_2d, dtype=np.int32)
                # Keep convex hull bounded strictly within image
                hull = cv2.convexHull(pts)
                hull_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(hull_mask, hull, 255)
                protection = np.maximum(protection, hull_mask)

        # 3. Construct editable region: expanded hair minus strict facial features
        editable = expanded_hair.copy()
        editable[protection > 0] = 0

        # Optional hole filling so scalp inpainting is contiguous
        if np.any(editable > 0):
            flood = editable.copy()
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flood, flood_mask, (0, 0), 255)
            inv_flood = cv2.bitwise_not(flood)
            editable = (editable | inv_flood)
            editable[protection > 0] = 0

        edit_mask_pil = Image.fromarray(editable, mode="L")
        protection_mask_pil = Image.fromarray(protection, mode="L")
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
        seed: Optional[int] = None,
    ) -> BaldConversionResult:
        t0 = time.perf_counter()
        w_orig, h_orig = target_image.size
        run_seed = seed if seed is not None else self.config.seed

        # Step 1: Construct edit and protection masks
        edit_mask_pil, protection_mask_pil, warnings = self.build_bald_edit_mask(
            target_image=target_image,
            target_hair_mask=target_hair_mask,
            target_segmentation_map=target_segmentation_map,
            face_landmarks=face_landmarks,
        )

        edit_np = np.asarray(edit_mask_pil)
        if np.count_nonzero(edit_np) == 0:
            warnings.append("Stage 1 edit mask is empty. Returning original target image as bald base.")
            return BaldConversionResult(
                bald_base=target_image.copy(),
                edit_mask=edit_mask_pil,
                protection_mask=protection_mask_pil,
                raw_generated=target_image.copy(),
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

        device = "cpu"
        if hasattr(pipe, "device") and isinstance(getattr(pipe, "device"), (str, torch.device)):
            device = pipe.device
        generator = torch.Generator(device=device).manual_seed(run_seed)

        # Set IP-Adapter scale to 0.0 for Stage 1 bald reconstruction (no hair conditioning)
        if hasattr(pipe, "set_ip_adapter_scale"):
            pipe.set_ip_adapter_scale(0.0)

        pipe_kwargs = {
            "prompt": self.config.prompt,
            "negative_prompt": self.config.negative_prompt,
            "image": pipe_input_img,
            "mask_image": pipe_mask_img,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "generator": generator,
        }
        # If pipeline has IP-Adapter loaded, provide ip_adapter_image to satisfy UNet cross-attention
        if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
            pipe_kwargs["ip_adapter_image"] = pipe_input_img

        diffusion_output = pipe(**pipe_kwargs)

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
            "dimensions": [w_orig, h_orig],
        }

        return BaldConversionResult(
            bald_base=bald_base,
            edit_mask=edit_mask_pil,
            protection_mask=protection_mask_pil,
            raw_generated=raw_generated,
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
