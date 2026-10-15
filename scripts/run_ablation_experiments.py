"""
Controlled Ablation Experiment Runner for HairTryOn.

Runs four controlled ablations with matching seed 42, 8 inference steps, scale 0.75:
1. Baseline: Legacy gray fade crop (dist/4.0) + Broad decontamination (core_hair == 0)
2. Crop-only fix: Revised antialiased crop with edge decontam + Broad decontamination
3. Ring-only fix: Legacy gray fade crop + Bounded ring decontamination
4. Combined fix: Revised antialiased crop + Bounded ring decontamination

Saves:
- IP-Adapter reference crops
- Alpha masks
- Decontaminated reference RGBs
- Aligned composite before diffusion
- Exact image and mask passed to Diffusers
- Raw diffusion output
- Final composited output
- Zoomed visual comparison image of hairline, temples, ears, and outer curls
- Pixel difference and color metrics
"""

import os
import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, ".")

# Point HF_HOME to D:/HFCache if present
if "HF_HOME" not in os.environ and Path("D:/HFCache").is_dir():
    os.environ["HF_HOME"] = "D:/HFCache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from app import segment_hair as app_segment_hair
from src.face_landmarks import detect_face_landmarks
from src.local_generative_transfer import (
    load_local_generative_pipeline,
    generate_local_hairstyle_transfer,
    prepare_reference_crop,
)
from src.hairstyle_alignment import align_reference_to_target


def run_ablations():
    base_dir = Path(__file__).resolve().parent.parent
    exp_src = base_dir / "outputs" / "experiments" / "exp_20261014_162340_42"
    out_base = base_dir / "outputs" / "experiments" / "ablation_comparison"
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Loading inputs from {exp_src}...")
    target_img = Image.open(exp_src / "target.png").convert("RGB")
    ref_img = Image.open(exp_src / "reference.png").convert("RGB")
    target_hair_mask = Image.open(exp_src / "hair_mask.png").convert("L")

    # Extract target face segmentation and landmarks
    print("Computing target segmentation and landmarks...")
    tgt_seg_res = app_segment_hair(target_img, return_seg_map=True)
    target_seg_map = tgt_seg_res[4]

    ref_mask, _, _, _ = app_segment_hair(ref_img)

    tgt_bgr = cv2.cvtColor(np.asarray(target_img), cv2.COLOR_RGB2BGR)
    ref_bgr = cv2.cvtColor(np.asarray(ref_img), cv2.COLOR_RGB2BGR)
    target_lms = detect_face_landmarks(tgt_bgr)
    ref_lms = detect_face_landmarks(ref_bgr)

    # Load SD inpainting pipeline once
    print("Loading local diffusion pipeline on CPU...")
    pipeline = load_local_generative_pipeline()

    ablations = [
        {
            "id": "1_baseline",
            "name": "Baseline (Legacy Gray Fade + Broad Decontam)",
            "legacy_crop": True,
            "bounded_decontam": False,
        },
        {
            "id": "2_crop_only",
            "name": "Revised Reference Crop Only",
            "legacy_crop": False,
            "bounded_decontam": False,
        },
        {
            "id": "3_ring_only",
            "name": "Bounded Edge Ring Only",
            "legacy_crop": True,
            "bounded_decontam": True,
        },
        {
            "id": "4_combined_fixes",
            "name": "Combined Fixes (Revised Crop + Bounded Ring)",
            "legacy_crop": False,
            "bounded_decontam": True,
        },
    ]

    results = {}
    seed = 42
    steps = 8
    scale = 0.75

    for abl in ablations:
        abl_id = abl["id"]
        abl_dir = out_base / abl_id
        abl_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n==========================================")
        print(f"Running Ablation: {abl['name']} ({abl_id})")
        print(f"==========================================")

        t0 = time.perf_counter()
        res = generate_local_hairstyle_transfer(
            target_image=target_img,
            reference_image=ref_img,
            target_hair_mask=target_hair_mask,
            reference_hair_mask=ref_mask,
            target_segmentation_map=target_seg_map,
            face_landmarks=target_lms,
            reference_landmarks=ref_lms,
            num_inference_steps=steps,
            ip_adapter_scale=scale,
            seed=seed,
            pipeline=pipeline,
            save_experiment=False,
            legacy_crop=abl["legacy_crop"],
            bounded_decontamination=abl["bounded_decontam"],
        )
        elapsed = time.perf_counter() - t0
        print(f"Ablation {abl_id} completed in {elapsed:.2f}s")

        # Save all individual artifacts
        res["reference_crop"].save(abl_dir / "ip_adapter_crop.png")
        res["composite_image"].save(abl_dir / "final_composite.png")
        res["raw_generated"].save(abl_dir / "raw_generated.png")
        res["edge_repair_mask"].save(abl_dir / "edge_repair_mask.png")
        res["edit_mask"].save(abl_dir / "broad_edit_mask.png")

        align_res = res["alignment_result"]
        if align_res:
            align_res["pasted_composite"].save(abl_dir / "aligned_composite_before_diffusion.png")
            if "decontaminated_reference_rgb" in align_res:
                Image.fromarray(align_res["decontaminated_reference_rgb"]).save(abl_dir / "decontaminated_reference.png")
            if "reference_hair_alpha" in align_res:
                Image.fromarray((align_res["reference_hair_alpha"] * 255).astype(np.uint8)).save(abl_dir / "reference_hair_alpha.png")
            if "unknown_region_mask" in align_res:
                Image.fromarray(align_res["unknown_region_mask"] * 255).save(abl_dir / "inpaint_unknown_mask.png")

        # Save metrics and config
        meta = {
            "id": abl_id,
            "name": abl["name"],
            "legacy_crop": abl["legacy_crop"],
            "bounded_decontamination": abl["bounded_decontam"],
            "seed": seed,
            "steps": steps,
            "scale": scale,
            "duration_seconds": round(elapsed, 2),
            "metrics": res["metrics"],
        }
        with open(abl_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        results[abl_id] = {
            "meta": meta,
            "composite": res["composite_image"],
            "crop": res["reference_crop"],
            "raw": res["raw_generated"],
            "pasted": align_res["pasted_composite"] if align_res else target_img,
            "edge_mask": res["edge_repair_mask"],
        }

    # Build side-by-side visual comparison and zoomed crops
    print("\nCreating comprehensive visual comparison collage...")
    w, h = 600, 600
    collage = Image.new("RGB", (w * 4, h + 300), color=(20, 20, 20))

    font = cv2.FONT_HERSHEY_SIMPLEX
    draw_canvas = np.asarray(collage).copy()

    for i, abl in enumerate(ablations):
        abl_id = abl["id"]
        comp_np = np.asarray(results[abl_id]["composite"])
        # Place full composite
        draw_canvas[50:h+50, i*w:(i+1)*w] = comp_np

        # Zoom into temple and hairline region: x: 150..350, y: 80..280 (200x200 window)
        zoom_window = comp_np[80:280, 150:350]
        zoom_resized = cv2.resize(zoom_window, (w, 240), interpolation=cv2.INTER_LANCZOS4)
        draw_canvas[h+60:h+300, i*w:(i+1)*w] = zoom_resized

        # Draw label headers
        title = abl["name"].split("(")[0].strip()
        cv2.putText(draw_canvas, f"{abl_id}: {title}", (i*w + 15, 35), font, 0.65, (255, 255, 255), 2)
        cv2.putText(draw_canvas, "Hairline & Temple Zoom (2x)", (i*w + 15, h + 45), font, 0.55, (200, 220, 255), 1)

    collage_img = Image.fromarray(draw_canvas)
    collage_img.save(out_base / "comparison_collage.png")
    print(f"Comparison collage saved to: {out_base / 'comparison_collage.png'}")

    # Build reference crops comparison collage
    crop_collage = Image.new("RGB", (512 * 2, 512), color=(0, 0, 0))
    crop_canvas = np.asarray(crop_collage).copy()
    crop_canvas[:, :512] = np.asarray(results["1_baseline"]["crop"])
    crop_canvas[:, 512:] = np.asarray(results["4_combined_fixes"]["crop"])
    cv2.putText(crop_canvas, "Baseline: Legacy 4px Gray Fade", (20, 40), font, 0.8, (0, 0, 255), 2)
    cv2.putText(crop_canvas, "Revised: Decontaminated & Antialiased", (512 + 20, 40), font, 0.8, (0, 255, 0), 2)
    Image.fromarray(crop_canvas).save(out_base / "reference_crops_comparison.png")
    print(f"Reference crops comparison saved to: {out_base / 'reference_crops_comparison.png'}")

    # Compute quantitative pixel differences between baseline and combined
    base_comp = np.asarray(results["1_baseline"]["composite"]).astype(float)
    comb_comp = np.asarray(results["4_combined_fixes"]["composite"]).astype(float)
    diff = np.abs(comb_comp - base_comp)
    print("\nQuantitative Ablation Differences:")
    print(f"Mean pixel diff across whole portrait: {diff.mean():.2f}")
    print(f"Max pixel diff: {diff.max():.2f}")
    edge_np = np.asarray(results["4_combined_fixes"]["edge_mask"]) > 128
    print(f"Mean pixel diff inside edge repair zone: {diff[edge_np].mean():.2f}")

    print("\nALL 4 ABLATION EXPERIMENTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_ablations()
