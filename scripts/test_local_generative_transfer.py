"""
Standalone proof of concept and validation script for local generative hairstyle transfer.

Workflow:
1. Preprocess & validate input images (target portrait & reference hairstyle).
2. Segment hair and facial features with ONNX ResNet18 face parser.
3. Detect facial landmarks and 3D head pose with MediaPipe.
4. Prepare reference hairstyle conditioning crop (with neutral background).
5. Build target edit mask with multi-layer facial feature protection.
6. Run local CPU inpainting using Stable Diffusion 1.5 + IP-Adapter Plus.
7. Feathered compositing with 100% bit-exact target preservation outside edit mask.
8. Quantitative evaluation, metadata serialization, and experiment persistence.
"""

from pathlib import Path
import sys
import os
import time
import argparse
from typing import Tuple

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if "HF_HOME" not in os.environ and Path("D:/HFCache").is_dir():
    os.environ["HF_HOME"] = "D:/HFCache"

# Ensure project root is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

from src.image_preprocessing import (
    load_and_validate_image,
    validate_target_portrait,
)
from src.face_landmarks import detect_face_landmarks
from src.local_generative_transfer import (
    load_local_generative_pipeline,
    prepare_reference_crop,
    build_target_edit_mask,
    generate_local_hairstyle_transfer,
    get_process_memory_mb,
    INFERENCE_PRESETS,
)


def get_segmentation(img_bgr: np.ndarray, session: ort.InferenceSession) -> Tuple[np.ndarray, np.ndarray]:
    """Run ONNX ResNet18 face parser and return (hair_mask, full_seg_map)."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    scale = min(512 / w, 512 / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(rgb, (nw, nh))
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    xo, yo = (512 - nw) // 2, (512 - nh) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized

    x = (canvas.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)

    outs = session.run(None, {session.get_inputs()[0].name: x})
    seg_512 = np.argmax(outs[0], axis=1)[0]
    seg_crop = seg_512[yo:yo + nh, xo:xo + nw]
    seg = cv2.resize(seg_crop.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    hair_mask = (seg == 17).astype(np.uint8) * 255
    return hair_mask, seg


def parse_args():
    parser = argparse.ArgumentParser(description="Test local generative hairstyle transfer.")
    parser.add_argument("--steps", type=int, default=10, help="Number of denoising steps (default: 10).")
    parser.add_argument("--scale", type=float, default=0.75, help="IP-Adapter conditioning scale (default: 0.75).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--target", type=str, default="assets/download.jpg", help="Path to target image.")
    parser.add_argument("--reference", type=str, default="assets/image.png", help="Path to reference image.")
    parser.add_argument("--alignment-only", action="store_true", help="Run geometric alignment only without diffusion (fast inspection).")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("HAIRTRYON LOCAL GENERATIVE ENGINE - STANDALONE VALIDATION")
    print("=" * 70)

    outputs_dir = BASE_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    target_path = BASE_DIR / args.target
    reference_path = BASE_DIR / args.reference

    if not target_path.exists() or not reference_path.exists():
        raise FileNotFoundError(f"Missing input assets: {target_path} or {reference_path}")

    # 1. Image loading & validation
    print("\n[Step 1/6] Loading and validating input portraits...")
    target_pil = load_and_validate_image(target_path)
    ref_pil = load_and_validate_image(reference_path)

    print(f"  Target:    {target_path.name} ({target_pil.width}x{target_pil.height}, {target_pil.mode})")
    print(f"  Reference: {reference_path.name} ({ref_pil.width}x{ref_pil.height}, {ref_pil.mode})")
    print(f"  Initial Process Memory: {get_process_memory_mb():.1f} MB RSS")

    # 2. Face landmark check
    print("\n[Step 2/6] Detecting face landmarks and head pose...")
    face_val = validate_target_portrait(target_pil)
    if face_val["warnings"]:
        for w in face_val["warnings"]:
            print(f"  Target Notice: {w}")
    target_landmarks = face_val.get("landmarks")

    ref_val = validate_target_portrait(ref_pil)
    if ref_val["warnings"]:
        for w in ref_val["warnings"]:
            print(f"  Reference Notice: {w}")
    ref_landmarks = ref_val.get("landmarks")

    if target_landmarks and target_landmarks.get("detected"):
        print(f"  Target Face:    Yaw {target_landmarks.get('yaw', 0.0):+.1f}°, Pitch {target_landmarks.get('pitch', 0.0):+.1f}°")
    if ref_landmarks and ref_landmarks.get("detected"):
        print(f"  Reference Face: Yaw {ref_landmarks.get('yaw', 0.0):+.1f}°, Pitch {ref_landmarks.get('pitch', 0.0):+.1f}°")

    # 3. ONNX Segmentation
    print("\n[Step 3/6] Running ONNX hair and face parsing on CPU...")
    session = ort.InferenceSession(str(BASE_DIR / "models" / "resnet18.onnx"), providers=["CPUExecutionProvider"])

    target_bgr = cv2.cvtColor(np.asarray(target_pil), cv2.COLOR_RGB2BGR)
    ref_bgr = cv2.cvtColor(np.asarray(ref_pil), cv2.COLOR_RGB2BGR)

    tgt_hair_mask, tgt_seg = get_segmentation(target_bgr, session)
    ref_hair_mask, ref_seg = get_segmentation(ref_bgr, session)

    tgt_mask_pil = Image.fromarray(tgt_hair_mask, mode="L")
    ref_mask_pil = Image.fromarray(ref_hair_mask, mode="L")

    tgt_hair_cov = 100.0 * np.count_nonzero(tgt_hair_mask) / tgt_hair_mask.size
    ref_hair_cov = 100.0 * np.count_nonzero(ref_hair_mask) / ref_hair_mask.size
    print(f"  Target hair coverage:    {tgt_hair_cov:.2f}%")
    print(f"  Reference hair coverage: {ref_hair_cov:.2f}%")

    pipeline = None
    if not args.alignment_only:
        # 4. Pipeline Loading
        print("\n[Step 4/6] Loading local Stable Diffusion + IP-Adapter Plus pipeline...", flush=True)
        load_start = time.perf_counter()
        pipeline = load_local_generative_pipeline(
            progress_callback=lambda msg: print(f"  > {msg}", flush=True),
        )
        load_time = time.perf_counter() - load_start
        print(f"  Pipeline ready in {load_time:.2f}s! Memory: {get_process_memory_mb():.1f} MB RSS", flush=True)
    else:
        print("\n[Step 4/6] Skipping diffusion pipeline loading (--alignment-only requested)...")

    # 5. Generative Hairstyle Transfer (or Alignment Preview)
    mode_str = "Alignment Only" if args.alignment_only else f"{args.steps} steps, scale={args.scale}, seed={args.seed}"
    print(f"\n[Step 5/6] Executing hairstyle pipeline ({mode_str})...", flush=True)
    res = generate_local_hairstyle_transfer(
        target_image=target_pil,
        reference_image=ref_pil,
        target_hair_mask=tgt_mask_pil,
        reference_hair_mask=ref_mask_pil,
        target_segmentation_map=tgt_seg,
        face_landmarks=target_landmarks,
        reference_landmarks=ref_landmarks,
        num_inference_steps=args.steps,
        ip_adapter_scale=args.scale,
        guidance_scale=7.5,
        seed=args.seed,
        cranial_expansion_px=30,
        use_alignment=True,
        alignment_only=args.alignment_only,
        save_experiment=not args.alignment_only,
        pipeline=pipeline,
        progress_callback=lambda msg: print(f"  > {msg}", flush=True),
    )

    align_info = res.get("alignment_result")
    if align_info:
        tf = align_info.get("transform_info", {})
        pose = align_info.get("pose_info", {})
        print("\n" + "=" * 70)
        print("GEOMETRIC ALIGNMENT DIAGNOSTICS")
        print("=" * 70)
        print(f"  Transform Mode:     {tf.get('mode', 'unknown')}")
        print(f"  Mirroring Applied:  {pose.get('recommend_mirror', False)}")
        print(f"  Scale Factor:       {tf.get('scale', 1.0):.3f}")
        print(f"  Rotation Angle:     {tf.get('angle_deg', 0.0):.2f}°")
        print(f"  Translation (tx,ty): ({tf.get('tx', 0.0):.1f}, {tf.get('ty', 0.0):.1f})")
        print(f"  Confidence Score:   {tf.get('confidence', 0.0):.2f}")
        if align_info.get("warnings"):
            for w in align_info["warnings"]:
                print(f"  Alignment Warning:  {w}")
        if align_info.get("preview_paths"):
            print("  Generated Previews:")
            for k, p in align_info["preview_paths"].items():
                print(f"    - {k}: {p}")
        print("=" * 70)

    out_path = outputs_dir / "local_generative_result.png"
    res["composite_image"].save(out_path)
    print(f"\n[Step 6/6] Inference Complete! Result saved to -> {out_path}")
    print(f"  Duration:     {res['duration_seconds']:.2f} seconds")
    print(f"  Peak RSS:     {res['peak_memory_mb']:.1f} MB")
    if res.get("experiment_dir"):
        print(f"  Experiment:   {res['experiment_dir']}")

    # Quantitative Evaluation Verification
    if not args.alignment_only:
        metrics = res["metrics"]
        print("\n" + "=" * 70)
        print("QUANTITATIVE EVALUATION REPORT")
        print("=" * 70)
        print(f"  Output Dimensions Match Target:       {metrics['dimension_match']}")
        print(f"  Max Pixel Diff Outside Edit Mask:     {metrics['max_outside_pixel_diff']} (Must be 0 for exact preservation)")
        print(f"  Outside Area Exact Preservation:      {metrics['outside_preservation_pct']}%")
        print(f"  Facial Features (Eyes/Nose/Mouth):    {'PROTECTED' if metrics['facial_features_protected'] else 'WARNING'}")
        print(f"  Edit Mask Coverage:                   {metrics['edit_mask_coverage_pct']}%")
        print(f"  Data Validity (No NaN / Inf / Blank): {not metrics['has_nan_or_inf'] and not metrics['is_blank_image']}")
        print("=" * 70)

        # Manual Review Checklist
        print("\nSTRUCTURED MANUAL REVIEW CHECKLIST:")
        for idx, q in enumerate(res["manual_review_questions"], 1):
            print(f"  [{idx}] {q['question']}")
        print("=" * 70)
    else:
        print("\n  [Alignment Inspection Mode] Skipping diffusion quantitative evaluation metrics.")


if __name__ == "__main__":
    main()
