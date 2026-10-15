"""
Transfer Evaluation and Experiment Tracking Module.

Provides:
1. Quantitative evaluation metrics (pixel preservation outside edit mask, feature protection, dimensions, data validity).
2. Experiment metadata recording and serialization under outputs/experiments/<experiment_id>/.
3. Structured manual review checklist for qualitative assessment.
"""

from pathlib import Path
import json
import time
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image


MANUAL_REVIEW_QUESTIONS = [
    {
        "id": "old_hair_removed",
        "question": "Was the old hairstyle removed without showing through underneath?",
    },
    {
        "id": "resembles_reference",
        "question": "Does the new hair resemble the reference hairstyle's texture, curls, or volume?",
    },
    {
        "id": "face_preserved",
        "question": "Are the target face, eyes, expression, skin, and beard completely preserved?",
    },
    {
        "id": "hairline_natural",
        "question": "Is the hairline transition at the forehead and temples natural?",
    },
    {
        "id": "edges_blended",
        "question": "Are the boundary edges softly blended without visible halos or hard cutoffs?",
    },
    {
        "id": "background_intact",
        "question": "Is the background outside the hair region 100% intact?",
    },
    {
        "id": "no_artifacts",
        "question": "Is the image free of floating hair patches, smudges, or extra ears?",
    },
]


def evaluate_transfer_metrics(
    target_image: Image.Image,
    composite_image: Image.Image,
    edit_mask: Image.Image,
    target_segmentation_map: Optional[np.ndarray] = None,
    duration_seconds: float = 0.0,
    peak_memory_mb: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate quantitative evaluation metrics on the hairstyle transfer result.
    """
    tgt_np = np.asarray(target_image.convert("RGB"), dtype=np.uint8)
    comp_np = np.asarray(composite_image.convert("RGB"), dtype=np.uint8)
    mask_np = np.asarray(edit_mask.convert("L"), dtype=np.uint8)

    w_tgt, h_tgt = target_image.size
    w_comp, h_comp = composite_image.size

    dimension_match = (w_tgt == w_comp) and (h_tgt == h_comp)

    # 1. Pixel Difference Outside Edit Mask
    outside_mask = mask_np == 0
    total_outside_pixels = int(np.count_nonzero(outside_mask))

    if total_outside_pixels > 0:
        abs_diff_outside = np.abs(tgt_np[outside_mask].astype(int) - comp_np[outside_mask].astype(int))
        max_diff_outside = int(np.max(abs_diff_outside))
        mean_diff_outside = float(np.mean(abs_diff_outside))
        exact_match_pixels = int(np.count_nonzero(abs_diff_outside == 0)) // 3
        outside_preservation_pct = float(100.0 * exact_match_pixels / total_outside_pixels)
    else:
        max_diff_outside = 0
        mean_diff_outside = 0.0
        outside_preservation_pct = 100.0

    # 2. Facial Features Protection Check
    features_protected = True
    features_max_diff = 0
    if target_segmentation_map is not None:
        seg = target_segmentation_map
        if seg.shape[:2] != (h_tgt, w_tgt):
            import cv2
            seg = cv2.resize(seg, (w_tgt, h_tgt), interpolation=cv2.INTER_NEAREST)
        # Check eyes (4, 5) and nose (10) and mouth (11, 12, 13)
        feat_region = np.isin(seg, [4, 5, 10, 11, 12, 13])
        if np.any(feat_region):
            feat_diff = np.abs(tgt_np[feat_region].astype(int) - comp_np[feat_region].astype(int))
            features_max_diff = int(np.max(feat_diff))
            features_protected = (features_max_diff == 0)

    # 3. Data Integrity & NaN / Inf checks
    comp_float = np.asarray(composite_image, dtype=np.float32)
    has_nan_or_inf = bool(np.isnan(comp_float).any() or np.isinf(comp_float).any())
    is_blank = bool(comp_np.std() < 1e-3)

    edit_mask_coverage_pct = float(100.0 * np.count_nonzero(mask_np > 128) / mask_np.size)

    return {
        "dimension_match": dimension_match,
        "max_outside_pixel_diff": max_diff_outside,
        "mean_outside_pixel_diff": round(mean_diff_outside, 4),
        "outside_preservation_pct": round(outside_preservation_pct, 2),
        "facial_features_protected": features_protected,
        "facial_features_max_diff": features_max_diff,
        "has_nan_or_inf": has_nan_or_inf,
        "is_blank_image": is_blank,
        "edit_mask_coverage_pct": round(edit_mask_coverage_pct, 2),
        "duration_seconds": round(duration_seconds, 2),
        "peak_memory_mb": round(peak_memory_mb, 1),
    }


def save_experiment_run(
    outputs_base_dir: Path,
    target_image: Image.Image,
    reference_image: Image.Image,
    hair_mask: Image.Image,
    edit_mask: Image.Image,
    composite_image: Image.Image,
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
    experiment_id: Optional[str] = None,
) -> Path:
    """
    Save complete experiment data under outputs/experiments/<experiment_id>/.

    Files written:
    - target.png
    - reference.png
    - hair_mask.png
    - edit_mask.png
    - generated.png
    - metadata.json
    - metrics.json

    Returns:
        Path to the created experiment directory.
    """
    if experiment_id is None:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        experiment_id = f"exp_{timestamp_str}_{metadata.get('seed', 42)}"

    exp_dir = Path(outputs_base_dir) / "experiments" / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save PNG assets
    target_image.save(exp_dir / "target.png")
    reference_image.save(exp_dir / "reference.png")
    hair_mask.save(exp_dir / "hair_mask.png")
    edit_mask.save(exp_dir / "edit_mask.png")
    composite_image.save(exp_dir / "generated.png")

    # Save JSON files
    with open(exp_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return exp_dir


def list_saved_experiments(outputs_base_dir: Path) -> list:
    """
    Scan outputs/experiments/ and return a list of saved experiment dictionaries,
    ordered by most recent first.
    """
    exp_base = Path(outputs_base_dir) / "experiments"
    if not exp_base.is_dir():
        return []

    experiments = []
    for d in exp_base.iterdir():
        if d.is_dir() and (d / "metadata.json").is_file():
            try:
                with open(d / "metadata.json", "r", encoding="utf-8") as f:
                    meta = json.load(f)
                metrics = {}
                if (d / "metrics.json").is_file():
                    with open(d / "metrics.json", "r", encoding="utf-8") as f:
                        metrics = json.load(f)

                experiments.append({
                    "id": d.name,
                    "dir": d,
                    "target_path": d / "target.png",
                    "reference_path": d / "reference.png",
                    "generated_path": d / "generated.png",
                    "edit_mask_path": d / "edit_mask.png",
                    "metadata": meta,
                    "metrics": metrics,
                    "mtime": d.stat().st_mtime,
                })
            except Exception:
                continue

    # Sort descending by modification time
    experiments.sort(key=lambda x: x["mtime"], reverse=True)
    return experiments
