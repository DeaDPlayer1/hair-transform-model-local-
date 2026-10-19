import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, ".")

# Setup local Hugging Face cache
if "HF_HOME" not in os.environ and Path("D:/HFCache").is_dir():
    os.environ["HF_HOME"] = "D:/HFCache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import cv2
from PIL import Image

from src.hair_segmentation import segment_hair as app_segment_hair
from src.face_landmarks import detect_face_landmarks
from src.local_generative_transfer import load_local_generative_pipeline
from src.two_stage_transfer import generate_two_stage_hairstyle_transfer


def resolve_image_path(p_str: str) -> Path:
    """Robustly resolve image paths handling spaces, underscores, and relative/absolute paths."""
    p = Path(p_str)
    if p.is_file():
        return p.resolve()

    # Try matching space <-> underscore variations
    cand_space = Path(str(p).replace("_", " "))
    if cand_space.is_file():
        return cand_space.resolve()

    cand_under = Path(str(p).replace(" ", "_"))
    if cand_under.is_file():
        return cand_under.resolve()

    # Check if file exists inside assets/ directory as fallback
    assets_dir = Path("assets")
    if assets_dir.is_dir():
        cand_asset = assets_dir / p.name
        if cand_asset.is_file():
            return cand_asset.resolve()
        cand_asset_space = assets_dir / p.name.replace("_", " ")
        if cand_asset_space.is_file():
            return cand_asset_space.resolve()
        cand_asset_under = assets_dir / p.name.replace(" ", "_")
        if cand_asset_under.is_file():
            return cand_asset_under.resolve()

    return p


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run complete Two-Stage Hairstyle Transfer (Stage 1: Bald Conversion, Stage 2: Hair Transfer)."
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        required=True,
        help="Path to the target portrait image (e.g. assets/target.jpg or path/to/target.png)",
    )
    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        required=True,
        help="Path to the hairstyle reference image (e.g. 'assets/hair reference.jpg' or path/to/ref.jpg)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs/two_stage_runs",
        help="Output directory (default: outputs/two_stage_runs)",
    )
    parser.add_argument(
        "--stage1-backend",
        type=str,
        choices=["auto", "sd-inpaint", "onnx"],
        default="auto",
        help="Stage 1 Bald Converter backend (default: auto)",
    )
    parser.add_argument(
        "--stage1-steps",
        type=int,
        default=8,
        help="Stage 1 diffusion inference steps (default: 8)",
    )
    parser.add_argument(
        "--stage2-steps",
        type=int,
        default=8,
        help="Stage 2 diffusion inference steps (default: 8)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.75,
        help="IP-Adapter hairstyle feature conditioning scale (default: 0.75)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (default: 7.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42)",
    )
    parser.add_argument(
        "--controlnet",
        type=str,
        choices=["auto", "on", "off"],
        default="auto",
        help="ControlNet spatial guidance (default: auto; disabled if no local ControlNet weights found)",
    )
    parser.add_argument(
        "--no-save-experiment",
        action="store_true",
        help="Disable saving experiment artifacts to disk",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    target_path = resolve_image_path(args.target)
    ref_path = resolve_image_path(args.reference)

    if not target_path.is_file():
        print(f"\n[ERROR] Target image file not found: '{args.target}' (resolved: '{target_path}')")
        sys.exit(1)
    if not ref_path.is_file():
        print(f"\n[ERROR] Reference image file not found: '{args.reference}' (resolved: '{ref_path}')")
        sys.exit(1)

    print("\n========================================================")
    print("  HairTryOn — Two-Stage Hairstyle Transfer Pipeline")
    print("========================================================")
    print(f"Target path:         {target_path}")
    print(f"Reference path:      {ref_path}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Stage 1 backend:     {args.stage1_backend}")
    print(f"Stage 1 steps:       {args.stage1_steps}")
    print(f"Stage 2 steps:       {args.stage2_steps}")
    print(f"IP-Adapter scale:    {args.scale}")
    print(f"Guidance scale:      {args.guidance_scale}")
    print(f"Seed:                {args.seed}")
    print(f"ControlNet:          {args.controlnet}\n")

    # 1. Load images and check dimensions
    print("[1/5] Loading target and reference images...")
    try:
        target_img = Image.open(target_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Failed to read input images: {e}")
        sys.exit(1)

    w_t, h_t = target_img.size
    w_r, h_r = ref_img.size
    print(f"      Target dimensions:    {w_t}x{h_t}")
    print(f"      Reference dimensions: {w_r}x{h_r}")

    # 2. Run segmentation and hair mask analysis
    print("[2/5] Segmenting hair masks and facial landmarks...")
    t_seg_res = app_segment_hair(target_img, return_seg_map=True)
    target_hair_mask = t_seg_res[0]
    target_seg_map = t_seg_res[4]

    r_seg_res = app_segment_hair(ref_img, return_seg_map=True)
    ref_hair_mask = r_seg_res[0]

    tgt_hair_np = np.asarray(target_hair_mask.convert("L"))
    tgt_cov = (np.count_nonzero(tgt_hair_np > 128) / (w_t * h_t)) * 100.0
    print(f"      Target hair coverage:    {tgt_cov:.2f}% ({np.count_nonzero(tgt_hair_np > 128)} px)")

    ref_hair_np = np.asarray(ref_hair_mask.convert("L"))
    ref_cov = (np.count_nonzero(ref_hair_np > 128) / (w_r * h_r)) * 100.0
    print(f"      Reference hair coverage: {ref_cov:.2f}% ({np.count_nonzero(ref_hair_np > 128)} px)")

    tgt_bgr = cv2.cvtColor(np.asarray(target_img), cv2.COLOR_RGB2BGR)
    ref_bgr = cv2.cvtColor(np.asarray(ref_img), cv2.COLOR_RGB2BGR)
    target_lms = detect_face_landmarks(tgt_bgr)
    ref_lms = detect_face_landmarks(ref_bgr)
    print(f"      Target landmarks:    {'Detected' if target_lms.get('detected') else 'Not detected'}")
    print(f"      Reference landmarks: {'Detected' if ref_lms.get('detected') else 'Not detected'}")

    # 3. Load generative pipeline once
    print("[3/5] Loading generative pipeline (SD 1.5 + IP-Adapter Plus)...")
    t_pipe = time.perf_counter()
    pipeline = load_local_generative_pipeline()
    print(f"      Pipeline ready in {time.perf_counter() - t_pipe:.1f}s")

    # 4. Execute Two-Stage Generation
    print("\n[4/5] Executing Two-Stage Hairstyle Transfer...")
    print(f"      -> Running Stage 1: Bald conversion (backend={args.stage1_backend}, steps={args.stage1_steps})...")
    res = generate_two_stage_hairstyle_transfer(
        target_image=target_img,
        reference_image=ref_img,
        pipeline=pipeline,
        target_hair_mask=target_hair_mask,
        reference_hair_mask=ref_hair_mask,
        target_segmentation_map=target_seg_map,
        face_landmarks=target_lms,
        reference_landmarks=ref_lms,
        stage1_backend=args.stage1_backend,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        ip_adapter_scale=args.scale,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        use_controlnet=args.controlnet,
        save_experiment=(not args.no_save_experiment),
        output_dir=args.output_dir,
    )

    metrics = res["metrics"]
    metadata = res["metadata"]

    print("\n[5/5] Two-Stage Generation Complete!")
    print(f"      Stage 1 Duration:    {metrics['stage1_duration_seconds']}s")
    print(f"      Stage 2 Duration:    {metrics['stage2_duration_seconds']}s")
    print(f"      Total Duration:      {metrics['total_duration_seconds']}s")
    print(f"      Stage 1 Backend:     {metadata['stage1_backend']}")

    print("\nKey Validation Metrics:")
    print(f"  • Dimensions Match:                  {metrics['dimension_match']} ({w_t}x{h_t})")
    print(f"  • Outside Pixel Preservation:        {metrics['outside_pixel_preservation_pct']}% (max diff: {metrics['max_outside_pixel_diff']})")
    print(f"  • Identity Protection Leaks:         {metrics['identity_protection_leak_count']} px (MUST BE 0)")
    print(f"  • Stage 1 Edit Mask Coverage:        {metrics['stage1_edit_mask_coverage_pct']}%")
    print(f"  • Stage 2 Broad Envelope Coverage:   {metrics['broad_envelope_coverage_pct']}%")
    print(f"  • Donor Core Curls Preserved:        {metrics['donor_core_pixel_count']} px")

    if res.get("experiment_dir"):
        print(f"\nSaved Run Artifacts Directory: {res['experiment_dir']}")
        print(f"  • Final Output:         {Path(res['experiment_dir']) / 'final_composite.png'}")
        print(f"  • 5-Panel Collage:      {Path(res['experiment_dir']) / 'comparison_collage.png'}")
        print(f"  • Stage 1 Bald Base:    {Path(res['experiment_dir']) / 'bald_base.png'}")
        print(f"  • Pre-Diffusion Comp:   {Path(res['experiment_dir']) / 'aligned_composite_before_diffusion.png'}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
