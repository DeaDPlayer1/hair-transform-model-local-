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
from src.bald_converter import (
    BaldConverterConfig,
    create_bald_converter,
    compute_inpaint_dimensions,
)
from src.local_generative_transfer import load_local_generative_pipeline
from src.two_stage_transfer import generate_two_stage_hairstyle_transfer


def resolve_image_path(p_str: str) -> Path:
    """Robustly resolve image paths handling spaces, underscores, extension variations (.jpg/.png), and assets/ lookups."""
    if not p_str:
        return Path(p_str)

    p = Path(p_str)
    if p.is_file():
        return p.resolve()

    IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG"]
    cand_dirs = [p.parent, Path("."), Path("assets")]
    cand_stems = [
        p.stem,
        p.stem.replace("_", " "),
        p.stem.replace(" ", "_"),
        p.name,
        p.name.replace("_", " "),
        p.name.replace(" ", "_"),
    ]

    # 1. Direct suffix variants in candidate directories
    for d in cand_dirs:
        if not d.is_dir():
            continue
        for stem in cand_stems:
            for ext in IMAGE_EXTS:
                cand = d / f"{stem}{ext}"
                if cand.is_file():
                    return cand.resolve()

    # 2. Case-insensitive and stem-matching scan
    for d in cand_dirs:
        if not d.is_dir():
            continue
        try:
            for item in d.iterdir():
                if item.is_file():
                    if item.stem.lower() in [s.lower() for s in cand_stems]:
                        return item.resolve()
                    if item.name.lower() in [s.lower() for s in cand_stems]:
                        return item.resolve()
        except Exception:
            pass

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
        default=None,
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
        "--stage1-only",
        action="store_true",
        help="Run Stage 1 only (bald conversion and scalp generation) and exit before Stage 2",
    )
    parser.add_argument(
        "--bald-base",
        type=str,
        default=None,
        help="Optional path to existing bald base image to skip Stage 1 bald conversion (e.g. outputs/.../bald_base.png)",
    )
    parser.add_argument(
        "--headwear-mask",
        type=str,
        default=None,
        help="Optional path to custom or manual headwear / stage 1 removal mask",
    )
    parser.add_argument(
        "--no-detect-headwear",
        action="store_true",
        help="Disable automatic headwear detector fallback",
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
        default=12,
        help="Stage 1 diffusion inference steps (default: 12)",
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
    ref_path = resolve_image_path(args.reference) if args.reference else None

    def _print_available_assets():
        assets_p = Path("assets")
        if assets_p.is_dir():
            files = [f.name for f in assets_p.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
            print(f"Available images in assets/: {', '.join(sorted(files))}")

    if not target_path.is_file():
        print(f"\n[ERROR] Target image file not found: '{args.target}' (resolved: '{target_path}')")
        _print_available_assets()
        sys.exit(1)

    if not args.stage1_only and (ref_path is None or not ref_path.is_file()):
        print(f"\n[ERROR] Reference image required for full two-stage run: '{args.reference}'")
        _print_available_assets()
        sys.exit(1)

    print("\n========================================================")
    print("  HairTryOn — Two-Stage Hairstyle Transfer Pipeline")
    print("========================================================")
    print("Integrated Core Models & Architecture:")
    print("  • Segmentation:      ResNet18 Face Parsing (ONNX Runtime CPU)")
    print("  • Generative Model:  Stable Diffusion 1.5 Inpainting (Diffusers)")
    print("  • Style Transfer:    IP-Adapter Plus (ViT-H/14 Cross-Attention)")
    print("  • Stage 1 Core:      BaldConverter (src/bald_converter.py)")
    print("  • Stage 2 Core:      TwoStageTransfer (src/two_stage_transfer.py)")
    print("--------------------------------------------------------")
    print(f"Target path:         {target_path}")
    print(f"Reference path:      {ref_path if ref_path else '(None - Stage 1 only)'}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Stage 1 only:        {args.stage1_only}")
    print(f"Headwear mask:       {args.headwear_mask}")
    print(f"Detect headwear:     {not args.no_detect_headwear}")
    print(f"Stage 1 backend:     {args.stage1_backend}")
    print(f"Stage 1 steps:       {args.stage1_steps}")
    if not args.stage1_only:
        print(f"Stage 2 steps:       {args.stage2_steps}")
        print(f"IP-Adapter scale:    {args.scale}")
    print(f"Guidance scale:      {args.guidance_scale}")
    print(f"Seed:                {args.seed}")
    print(f"ControlNet:          {args.controlnet}\n")

    # 1. Load images and check dimensions
    print("[1/5] Loading target image...")
    try:
        from src.image_preprocessing import load_and_validate_image

        target_img = load_and_validate_image(target_path)
        ref_img = load_and_validate_image(ref_path) if ref_path else None
    except Exception as e:
        print(f"[ERROR] Failed to read input image: {e}")
        sys.exit(1)

    w_t, h_t = target_img.size
    print(f"      Target dimensions:    {w_t}x{h_t}")
    if ref_img:
        w_r, h_r = ref_img.size
        print(f"      Reference dimensions: {w_r}x{h_r}")

    # 2. Run segmentation and hair mask analysis
    print("[2/5] Segmenting hair masks and facial landmarks...")
    t_seg_res = app_segment_hair(target_img, return_seg_map=True, return_hair_prob=True)
    target_hair_mask = t_seg_res[0]
    target_seg_map = t_seg_res[4]
    target_hair_prob = t_seg_res[5]

    ref_hair_mask = None
    ref_seg_map = None
    if ref_img:
        r_seg_res = app_segment_hair(ref_img, return_seg_map=True)
        ref_hair_mask = r_seg_res[0]
        ref_seg_map = r_seg_res[4]
        ref_hair_np = np.asarray(ref_hair_mask.convert("L"))
        ref_cov = (np.count_nonzero(ref_hair_np > 128) / (w_r * h_r)) * 100.0
        print(f"      Reference hair coverage: {ref_cov:.2f}% ({np.count_nonzero(ref_hair_np > 128)} px)")

    tgt_hair_np = np.asarray(target_hair_mask.convert("L"))
    tgt_cov = (np.count_nonzero(tgt_hair_np > 128) / (w_t * h_t)) * 100.0
    print(f"      Target hair coverage:    {tgt_cov:.2f}% ({np.count_nonzero(tgt_hair_np > 128)} px)")

    tgt_bgr = cv2.cvtColor(np.asarray(target_img), cv2.COLOR_RGB2BGR)
    target_lms = detect_face_landmarks(tgt_bgr)
    print(f"      Target landmarks:    {'Detected' if target_lms.get('detected') else 'Not detected'}")

    ref_lms = None
    if ref_img:
        ref_bgr = cv2.cvtColor(np.asarray(ref_img), cv2.COLOR_RGB2BGR)
        ref_lms = detect_face_landmarks(ref_bgr)
        print(f"      Reference landmarks: {'Detected' if ref_lms.get('detected') else 'Not detected'}")

    # 3. Load generative pipeline once
    print("[3/5] Loading generative pipeline (SD 1.5)...")
    t_pipe = time.perf_counter()
    pipeline = load_local_generative_pipeline()
    print(f"      Pipeline ready in {time.perf_counter() - t_pipe:.1f}s")

    # 4. Execute Two-Stage Generation
    print("\n[4/5] Executing Generation Pipeline...")
    bald_base_img = None
    if args.bald_base:
        b_path = resolve_image_path(args.bald_base)
        if b_path.is_file():
            print(f"      -> Reusing provided bald base: {b_path}")
            bald_base_img = load_and_validate_image(b_path)
        else:
            print(f"      [WARN] Bald base '{args.bald_base}' not found; running Stage 1.")

    if bald_base_img is None:
        print(f"      -> Running Stage 1: Bald conversion (backend={args.stage1_backend}, steps={args.stage1_steps})...")
    res = generate_two_stage_hairstyle_transfer(
        target_image=target_img,
        reference_image=ref_img,
        pipeline=pipeline,
        bald_base=bald_base_img,
        target_hair_mask=target_hair_mask,
        reference_hair_mask=ref_hair_mask,
        target_segmentation_map=target_seg_map,
        reference_segmentation_map=ref_seg_map,
        target_hair_prob=target_hair_prob,
        face_landmarks=target_lms,
        reference_landmarks=ref_lms,
        stage1_backend=args.stage1_backend,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        ip_adapter_scale=args.scale,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        use_controlnet=args.controlnet,
        stage1_only=args.stage1_only,
        headwear_mask=args.headwear_mask,
        detect_headwear=(not args.no_detect_headwear),
        save_experiment=(not args.no_save_experiment),
        output_dir=args.output_dir,
    )

    if args.stage1_only:
        print("\n[5/5] Stage 1 Bald Conversion Complete (stage1_only=True)!")
        if res.get("experiment_dir"):
            exp_d = Path(res["experiment_dir"])
            print(f"\nSaved Stage 1 Artifacts in: {exp_d}")
            print(f"  • Bald Base:           {exp_d / 'bald_base.png'}")
            print(f"  • Removal Overlay:     {exp_d / 'stage1_removal_overlay.png'}")
            print(f"  • Stage 1 Edit Mask:   {exp_d / 'stage1_edit_mask.png'}")
            print(f"  • Protection Mask:     {exp_d / 'stage1_protection_mask.png'}")
            print(f"  • Raw Generated Scalp: {exp_d / 'stage1_raw_generated.png'}")
        print("========================================================\n")
        return

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

    trans_info = metadata.get("transform_info", {})
    if trans_info.get("scale_corrected"):
        print(f"  • Scale Corrected (Yaw-Robust):      {trans_info.get('original_scale', 0.0):.2f}x -> {trans_info.get('scale', 0.0):.2f}x")
    if trans_info.get("pose_quality_flag") == "poor":
        print(f"  • Pose Quality Warning:              High pose difference detected (>45°)")

    if res.get("experiment_dir"):
        print(f"\nSaved Run Artifacts Directory: {res['experiment_dir']}")
        print(f"  • Final Output:         {Path(res['experiment_dir']) / 'final_composite.png'}")
        print(f"  • 5-Panel Collage:      {Path(res['experiment_dir']) / 'comparison_collage.png'}")
        print(f"  • Stage 1 Bald Base:    {Path(res['experiment_dir']) / 'bald_base.png'}")
        print(f"  • Stage 1 BG Diff:      {Path(res['experiment_dir']) / 'stage1_background_difference.png'}")
        print(f"  • Removal Overlay:      {Path(res['experiment_dir']) / 'stage1_removal_overlay.png'}")
        print(f"  • Donor Skin Excl:      {Path(res['experiment_dir']) / 'reference_skin_exclusion_mask.png'}")
        print(f"  • Pre-Diffusion Comp:   {Path(res['experiment_dir']) / 'aligned_composite_before_diffusion.png'}")
        print(f"  • Stage 2 Hairline ROI: {Path(res['experiment_dir']) / 'stage2_hairline_roi.png'}")
        print(f"  • Forehead Zoom Comp:   {Path(res['experiment_dir']) / 'compare_prediff_vs_final_forehead.png'}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
