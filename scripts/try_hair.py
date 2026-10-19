import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, ".")

# Use local cache if present
if "HF_HOME" not in os.environ and Path("D:/HFCache").is_dir():
    os.environ["HF_HOME"] = "D:/HFCache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import cv2
from PIL import Image

from src.hair_segmentation import segment_hair as app_segment_hair
from src.face_landmarks import detect_face_landmarks
from src.local_generative_transfer import (
    load_local_generative_pipeline,
    generate_local_hairstyle_transfer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local hairstyle transfer on any target image and hair reference image."
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        required=True,
        help="Path to the target portrait image (e.g. assets/target.jpg or path/to/your/photo.jpg)",
    )
    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        required=True,
        help="Path to the reference hairstyle image (e.g. assets/reference.jpg or path/to/hair.jpg)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs/custom_transfer",
        help="Directory to save the resulting images (default: outputs/custom_transfer)",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=8,
        help="Number of diffusion inference steps (default: 8; 6-10 is optimal for speed/quality on CPU)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.75,
        help="IP-Adapter conditioning scale (default: 0.75)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    target_path = Path(args.target)
    ref_path = Path(args.reference)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not target_path.is_file():
        print(f"Error: Target image file not found at '{target_path}'")
        sys.exit(1)
    if not ref_path.is_file():
        print(f"Error: Reference image file not found at '{ref_path}'")
        sys.exit(1)

    print(f"\n========================================================")
    print(f"  HairTryOn — Local Generative Transfer")
    print(f"========================================================")
    print(f"Target image:    {target_path}")
    print(f"Reference image: {ref_path}")
    print(f"Output folder:   {out_dir}")
    print(f"Steps:           {args.steps}")
    print(f"Seed:            {args.seed}")
    print(f"IP-Adapter scale:{args.scale}\n")

    print("[1/5] Loading images...")
    target_img = Image.open(target_path).convert("RGB")
    ref_img = Image.open(ref_path).convert("RGB")
    print(f"      Target size: {target_img.size}, Reference size: {ref_img.size}")

    print("[2/5] Running hair segmentation & face parsing...")
    target_seg_res = app_segment_hair(target_img, return_seg_map=True)
    target_hair_mask = target_seg_res[0]
    target_seg_map = target_seg_res[4]

    ref_seg_res = app_segment_hair(ref_img, return_seg_map=True)
    ref_hair_mask = ref_seg_res[0]

    print("[3/5] Detecting facial landmarks...")
    tgt_bgr = cv2.cvtColor(np.asarray(target_img), cv2.COLOR_RGB2BGR)
    ref_bgr = cv2.cvtColor(np.asarray(ref_img), cv2.COLOR_RGB2BGR)
    target_lms = detect_face_landmarks(tgt_bgr)
    ref_lms = detect_face_landmarks(ref_bgr)

    print("[4/5] Loading Stable Diffusion + IP-Adapter models...")
    pipeline = load_local_generative_pipeline()

    print(f"[5/5] Generating hairstyle transfer ({args.steps} steps)...")
    t0 = time.perf_counter()
    res = generate_local_hairstyle_transfer(
        target_image=target_img,
        reference_image=ref_img,
        target_hair_mask=target_hair_mask,
        reference_hair_mask=ref_hair_mask,
        target_segmentation_map=target_seg_map,
        face_landmarks=target_lms,
        reference_landmarks=ref_lms,
        num_inference_steps=args.steps,
        ip_adapter_scale=args.scale,
        seed=args.seed,
        pipeline=pipeline,
        save_experiment=False,
        legacy_crop=False,
        bounded_decontamination=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nGeneration completed in {elapsed:.1f} seconds!")

    # Save individual results
    final_comp = res["composite_image"]
    final_path = out_dir / "final_composite.png"
    final_comp.save(final_path)

    res["edge_repair_mask"].save(out_dir / "edge_repair_mask.png")
    res["edit_mask"].save(out_dir / "broad_edit_mask.png")

    align_res = res.get("alignment_result")
    if align_res and "pasted_composite" in align_res:
        align_res["pasted_composite"].save(out_dir / "aligned_composite_before_diffusion.png")

    # Generate side-by-side comparison collage: Target | Reference | Pre-Diffusion | Final
    h, w = target_img.size[1], target_img.size[0]
    pasted_pil = align_res["pasted_composite"] if (align_res and "pasted_composite" in align_res) else target_img
    ref_resized = ref_img.resize((w, h), Image.Resampling.LANCZOS)

    collage = np.hstack([
        np.asarray(target_img),
        np.asarray(ref_resized),
        np.asarray(pasted_pil),
        np.asarray(final_comp),
    ])
    collage_path = out_dir / "comparison_collage.png"
    Image.fromarray(collage).save(collage_path)

    print("\nResults saved:")
    print(f"  • Final result:       {final_path.resolve()}")
    print(f"  • 4-panel collage:    {collage_path.resolve()}")
    print(f"  • Alignment preview:  {(out_dir / 'aligned_composite_before_diffusion.png').resolve()}")
    print(f"  • Repair masks:       {(out_dir / 'edge_repair_mask.png').resolve()}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
