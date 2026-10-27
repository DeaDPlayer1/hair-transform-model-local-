"""
Stage 1 Generalization Campaign Runner.
Dynamically discovers every target image, constructs adaptive masks,
executes Stage 1 bald base conversion, evaluates Quality Gate standards,
and writes comprehensive reports and artifacts.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hair_segmentation import segment_hair
from src.face_landmarks import detect_face_landmarks
from src.headwear_detection import detect_headwear, create_removal_overlay
from src.bald_converter import (
    BaldConverterConfig,
    SDInpaintBaldConverter,
    compute_face_geometry,
    compute_inpaint_dimensions,
    evaluate_stage1_quality_gate,
)
from src.local_generative_transfer import load_local_generative_pipeline
from src.image_preprocessing import load_and_validate_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage1_generalization")

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif")


def discover_target_images(assets_dir: Path = Path("assets")) -> List[Path]:
    """Dynamically discover all valid target images in assets directory."""
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"Assets directory not found at {assets_dir.resolve()}")
    targets = [
        p
        for p in sorted(assets_dir.iterdir())
        if p.is_file()
        and p.name.lower().startswith("target")
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return targets


def run_campaign(
    assets_dir: Path = Path("assets"),
    output_root: Path = Path("outputs/stage1_generalization"),
    num_inference_steps: int = 12,
    seed: int = 42,
    guidance_scale: float = 7.5,
) -> Path:
    """Run full Stage 1 generalization campaign on all discovered target images."""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    targets = discover_target_images(assets_dir)
    logger.info(f"Discovered {len(targets)} target images: {[t.name for t in targets]}")

    # =========================================================
    # Phase 1: Pre-flight Inventory & Mask Analysis
    # =========================================================
    logger.info("Starting Phase 1: Pre-flight Dataset Inventory & Mask Construction...")
    inventory: List[Dict[str, Any]] = []

    for t_path in targets:
        t_rgb = load_and_validate_image(t_path)
        w, h = t_rgb.size
        mode = t_rgb.mode
        fmt = t_path.suffix.lstrip(".").upper()
        file_bytes = t_path.stat().st_size

        # Hair segmentation
        hair_mask_pil, _, seg_elapsed, hair_cov, seg_map = segment_hair(t_rgb, return_seg_map=True)
        hair_px = int(np.count_nonzero(np.asarray(hair_mask_pil) > 128))

        # Face landmarks
        lm = detect_face_landmarks(t_rgb)
        geom = compute_face_geometry(lm, w, h)

        # Headwear detection
        hw_res = detect_headwear(t_rgb, seg_map=seg_map, face_landmarks=lm, hair_mask=hair_mask_pil)

        inv_entry = {
            "filename": t_path.name,
            "path": str(t_path.resolve()),
            "format": fmt,
            "dimensions": [w, h],
            "mode": mode,
            "file_size_bytes": file_bytes,
            "landmarks_detected": geom["detected"],
            "inter_ocular_dist_px": round(geom["inter_ocular_dist"], 2),
            "face_scale_px": round(geom["face_scale"], 2),
            "head_pose": {
                "yaw": round(geom["yaw"], 2),
                "pitch": round(geom["pitch"], 2),
                "roll": round(geom["roll"], 2),
            },
            "hair_pixels": hair_px,
            "hair_coverage_pct": round(hair_cov, 2),
            "headwear_detected": hw_res.detected,
            "headwear_source": hw_res.source,
            "headwear_pixels": hw_res.pixel_count,
            "headwear_coverage_pct": round(hw_res.coverage_pct, 4),
            "classes_present": [int(c) for c in np.unique(seg_map)],
        }
        inventory.append(inv_entry)

    # Save dataset_inventory.json
    with (run_dir / "dataset_inventory.json").open("w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)
    logger.info("Saved dataset_inventory.json")

    # =========================================================
    # Phase 2: Load Model Pipeline Once
    # =========================================================
    logger.info("Loading Stable Diffusion generative pipeline...")
    shared_pipeline = load_local_generative_pipeline()

    # =========================================================
    # Phase 3: Stage 1 Inpainting Generation & Quality Gate
    # =========================================================
    test_matrix_rows: List[Dict[str, Any]] = []
    issue_logs: List[str] = []

    s1_cfg = BaldConverterConfig(
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    converter = SDInpaintBaldConverter(s1_cfg, pipeline=shared_pipeline)

    for idx, t_path in enumerate(targets, 1):
        target_name = t_path.stem
        t_out_dir = run_dir / target_name
        t_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{idx}/{len(targets)}] Processing {t_path.name}...")

        target_rgb = load_and_validate_image(t_path)
        w_orig, h_orig = target_rgb.size

        # Segmentation & landmarks
        hair_mask_pil, _, _, _, seg_map = segment_hair(target_rgb, return_seg_map=True)
        lm = detect_face_landmarks(target_rgb)

        # Run Stage 1 Bald Conversion
        t0 = time.perf_counter()
        result = converter.convert(
            target_image=target_img,
            target_hair_mask=hair_mask_pil,
            target_segmentation_map=seg_map,
            face_landmarks=lm,
            seed=seed,
        )
        elapsed = time.perf_counter() - t0

        # Save all canonical Section 5 artifacts:
        target_img.save(t_out_dir / "target_original.png")
        hair_mask_pil.save(t_out_dir / "target_hair_mask.png")
        if result.headwear_mask:
            result.headwear_mask.save(t_out_dir / "headwear_mask.png")
        else:
            Image.new("L", (w_orig, h_orig), 0).save(t_out_dir / "headwear_mask.png")

        result.protection_mask.save(t_out_dir / "strict_identity_protection.png")
        result.edit_mask.save(t_out_dir / "stage1_edit_mask.png")
        if result.removal_overlay:
            result.removal_overlay.save(t_out_dir / "stage1_removal_overlay.png")
        if result.raw_generated:
            result.raw_generated.save(t_out_dir / "stage1_raw_generated.png")
        result.bald_base.save(t_out_dir / "bald_base.png")

        # Create side-by-side comparison panel
        # [Original, Removal Overlay, Bald Base]
        comp_orig = target_rgb.resize((400, int(round(400 * h_orig / w_orig))), Image.Resampling.BILINEAR)
        comp_overlay = result.removal_overlay.convert("RGB").resize(comp_orig.size, Image.Resampling.BILINEAR)
        comp_bald = result.bald_base.convert("RGB").resize(comp_orig.size, Image.Resampling.BILINEAR)
        comp_panel = np.hstack([np.asarray(comp_orig), np.asarray(comp_overlay), np.asarray(comp_bald)])
        Image.fromarray(comp_panel).save(t_out_dir / "comparison_panel.png")

        # Metrics and warnings
        with (t_out_dir / "stage1_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(result.metadata, f, indent=2)
        with (t_out_dir / "warnings.json").open("w", encoding="utf-8") as f:
            json.dump(result.warnings, f, indent=2)

        qg = result.metadata.get("quality_gate", {})
        status = qg.get("status", "UNKNOWN")

        matrix_row = {
            "target": t_path.name,
            "dimensions": f"{w_orig}x{h_orig}",
            "mode": target_img.mode,
            "face_scale_px": result.metadata.get("face_scale", 0.0),
            "edit_coverage_pct": qg.get("edit_coverage_pct", 0.0),
            "preservation_pct": qg.get("preservation_pct", 0.0),
            "identity_leak_count": qg.get("identity_leak_count", 0),
            "headwear_detected": result.headwear_detected,
            "quality_status": status,
            "elapsed_seconds": round(elapsed, 2),
            "reasons": "; ".join(qg.get("failure_reasons", []) + qg.get("review_reasons", [])),
        }
        test_matrix_rows.append(matrix_row)

        if status != "PASSED":
            issue_logs.append(
                f"- **{t_path.name}**: Status={status}. Reasons: {matrix_row['reasons']}"
            )
        logger.info(f"  -> {t_path.name}: Quality Status = {status} ({elapsed:.1f}s)")

    # Save stage1_test_matrix.csv
    csv_path = run_dir / "stage1_test_matrix.csv"
    if test_matrix_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(test_matrix_rows[0].keys()))
            writer.writeheader()
            writer.writerows(test_matrix_rows)
        logger.info(f"Saved {csv_path}")

    # Save issue_log.md
    with (run_dir / "issue_log.md").open("w", encoding="utf-8") as f:
        f.write("# Stage 1 Generalization Campaign Issue Log\n\n")
        f.write(f"Run Directory: `{run_dir.resolve()}`\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        if issue_logs:
            f.write("## Flagged Items\n\n")
            f.write("\n".join(issue_logs) + "\n")
        else:
            f.write("## Status\n\nNo blocking issues detected across all discovered targets.\n")

    # Save code_change_log.md
    with (run_dir / "code_change_log.md").open("w", encoding="utf-8") as f:
        f.write("# Stage 1 Code Change Log\n\n")
        f.write("### 1. Dynamic Landmark-Driven Face Protection\n")
        f.write("- Replaced rigid horizontal `min_brow_y` threshold with a pose-adaptive lower-face polygon (`jaw_indices + brow_arch_indices`).\n")
        f.write("- Adapts dynamically to roll tilt and head pitch across all targets.\n\n")
        f.write("### 2. Anatomical Scale-Adaptive Morphology\n")
        f.write("- Replaced fixed pixel kernels with face-scale-derived morphology.\n")
        f.write("- Expansion, hole closing, and eyebrow safety dilations scale with inter-ocular distance and head scale.\n\n")
        f.write("### 3. Aspect-Ratio Preserving UNet Inpainting\n")
        f.write("- Implemented `compute_inpaint_dimensions()` to ensure multiples of 8 without squashing or stretching non-square portraits.\n\n")
        f.write("### 4. RGBA and Alpha Channel Preservation\n")
        f.write("- Handles RGBA inputs (e.g. `target1.png`) and restores original alpha channel in final output.\n\n")
        f.write("### 5. Automated Quality Gate\n")
        f.write("- Evaluates 100.00% bit-exact outside preservation, zero protection leak, and scalp plausibility.\n")

    # Save final_report.md
    passed_count = sum(1 for r in test_matrix_rows if r["quality_status"] == "PASSED")
    review_count = sum(1 for r in test_matrix_rows if r["quality_status"] == "NEEDS_VISUAL_REVIEW")
    failed_count = sum(1 for r in test_matrix_rows if r["quality_status"] == "FAILED")

    with (run_dir / "final_report.md").open("w", encoding="utf-8") as f:
        f.write("# Stage 1 Generalization Campaign - Final Report\n\n")
        f.write(f"**Execution Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Targets Evaluated**: {len(targets)}\n")
        f.write(f"**Quality Gate Results**: {passed_count} Passed, {review_count} Needs Visual Review, {failed_count} Failed\n\n")
        f.write("## Test Matrix Summary\n\n")
        f.write("| Target | Dimensions | Mode | Scale (px) | Edit Cov % | Preservation % | Status |\n")
        f.write("|--------|------------|------|------------|------------|----------------|--------|\n")
        for r in test_matrix_rows:
            f.write(
                f"| `{r['target']}` | {r['dimensions']} | {r['mode']} | {r['face_scale_px']} | "
                f"{r['edit_coverage_pct']}% | {r['preservation_pct']}% | **{r['quality_status']}** |\n"
            )
        f.write("\n## Per-Target Observations\n\n")
        for r in test_matrix_rows:
            f.write(f"### {r['target']}\n")
            f.write(f"- Status: `{r['quality_status']}`\n")
            f.write(f"- Dimensions: {r['dimensions']}, Mode: {r['mode']}\n")
            f.write(f"- Edit Mask Coverage: {r['edit_coverage_pct']}%\n")
            f.write(f"- Outside Preservation: {r['preservation_pct']}%\n")
            f.write(f"- Notes: {r['reasons'] or 'All automated checks passed cleanly.'}\n\n")

    logger.info(f"Campaign complete! Results saved in {run_dir}")
    return run_dir


if __name__ == "__main__":
    run_campaign()
