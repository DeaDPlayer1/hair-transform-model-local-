"""
test_final.py — Interactive & CLI Hairstyle Transfer Runner for HairTryOn.

Allows choosing any target and hairstyle reference from `assets/`
without manually editing code. Supports fuzzy/normalized filename matching,
interactive mode, listing available assets, and parameter overrides.

Delegates execution to the validated two-stage hairstyle transfer pipeline:
  Stage 1: Bald Conversion (Scalp reconstruction)
  Stage 2: Reference Hairstyle Transfer (IP-Adapter conditioning + alignment + edge refinement)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
DEFAULT_ASSETS_DIR = Path("assets")
DEFAULT_OUTPUT_DIR = Path("outputs/two_stage_runs")


def normalize_token(s: str) -> str:
    """Normalize a filename or search token by lowercasing and removing punctuation/whitespace."""
    return re.sub(r"[\s_\-\.]+", "", s.lower())


def scan_assets(assets_dir: Path = DEFAULT_ASSETS_DIR) -> Dict[str, List[Path]]:
    """Scan assets directory and classify files into targets, references, and others."""
    targets: List[Path] = []
    references: List[Path] = []
    others: List[Path] = []

    if not assets_dir.is_dir():
        return {"targets": targets, "references": references, "others": others}

    for item in sorted(assets_dir.iterdir()):
        if not item.is_file():
            continue
        if item.suffix.lower() not in VALID_EXTENSIONS:
            continue

        norm = normalize_token(item.stem)
        if norm.startswith("target"):
            targets.append(item)
        elif norm.startswith("reference"):
            references.append(item)
        else:
            others.append(item)

    return {"targets": targets, "references": references, "others": others}


def print_asset_list(assets_dir: Path = DEFAULT_ASSETS_DIR):
    """Print an attractive summary table of all discovered images in assets/."""
    categorized = scan_assets(assets_dir)
    targets = categorized["targets"]
    references = categorized["references"]
    others = categorized["others"]

    total = len(targets) + len(references) + len(others)
    print("\n" + "=" * 70)
    print(f"  Available Images in: {assets_dir.resolve()} ({total} total)")
    print("=" * 70)

    def print_group(title: str, items: List[Path]):
        print(f"\n  [{title}] ({len(items)} found):")
        if not items:
            print("    (none)")
            return
        for idx, p in enumerate(items, start=1):
            dim_str = ""
            if Image:
                try:
                    with Image.open(p) as img:
                        dim_str = f"[{img.width}x{img.height}, {img.mode}]"
                except Exception:
                    dim_str = "[unreadable]"
            size_kb = p.stat().st_size / 1024.0
            print(f"    {idx:2d}. {p.name:<25} {dim_str:<18} ({size_kb:6.1f} KB)")

    print_group("Target Portraits (Faces to try hair on)", targets)
    print_group("Hairstyle References (Hairstyles to transfer)", references)
    if others:
        print_group("Other Images", others)

    print("\n" + "-" * 70)
    print("  You can reference any image by full filename, stem (e.g. 'target2'),")
    print("  or normalized name with spaces/underscores (e.g. 'target 2.jpg').")
    print("-" * 70 + "\n")


def match_image(
    query: str,
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    role_hint: Optional[str] = None,
) -> Optional[Path]:
    """
    Find matching image in assets directory using exact, stem, normalized, or fuzzy matching.
    """
    raw_path = Path(query)
    if raw_path.is_file():
        return raw_path.resolve()

    if not assets_dir.is_dir():
        return None

    # Check direct child
    direct_child = assets_dir / query
    if direct_child.is_file():
        return direct_child.resolve()

    # Collect all valid image candidates
    all_files = [
        f for f in assets_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]

    q_norm = normalize_token(query)
    q_stem_norm = normalize_token(raw_path.stem)

    # 1. Exact case-insensitive filename match
    for f in all_files:
        if f.name.lower() == query.lower():
            return f.resolve()

    # 2. Exact case-insensitive stem match
    for f in all_files:
        if f.stem.lower() == raw_path.stem.lower():
            return f.resolve()

    # 3. Normalized string match on filename or stem (e.g. 'target 2.jpg' -> 'target2')
    for f in all_files:
        if normalize_token(f.name) == q_norm or normalize_token(f.stem) == q_stem_norm:
            return f.resolve()

    # 4. If query is just a number (e.g. '3') and role_hint is given ('target' -> 'target3')
    if query.isdigit() and role_hint:
        candidate_name = f"{role_hint}{query}"
        for f in all_files:
            if normalize_token(f.stem) == candidate_name:
                return f.resolve()

    # 5. Prefix match
    matches = [f for f in all_files if normalize_token(f.stem).startswith(q_stem_norm)]
    if len(matches) == 1:
        return matches[0].resolve()

    # 6. Substring match
    substring_matches = [f for f in all_files if q_stem_norm in normalize_token(f.stem)]
    if len(substring_matches) == 1:
        return substring_matches[0].resolve()

    return None


def run_pipeline(
    target_path: Path,
    reference_path: Optional[Path] = None,
    stage1_steps: int = 12,
    stage2_steps: int = 8,
    scale: float = 0.75,
    guidance_scale: float = 7.5,
    seed: int = 42,
    stage1_backend: str = "auto",
    controlnet: str = "auto",
    stage1_only: bool = False,
    headwear_mask: Optional[str] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> int:
    """Invoke scripts/try_hair_two_stage.py with the resolved arguments."""
    script_path = Path("scripts/try_hair_two_stage.py")
    if not script_path.is_file():
        print(f"[ERROR] Two-stage pipeline script not found at {script_path.resolve()}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        str(script_path),
        "--target",
        str(target_path),
        "--stage1-steps",
        str(stage1_steps),
        "--stage2-steps",
        str(stage2_steps),
        "--scale",
        str(scale),
        "--guidance-scale",
        str(guidance_scale),
        "--seed",
        str(seed),
        "--stage1-backend",
        str(stage1_backend),
        "--controlnet",
        str(controlnet),
        "--output-dir",
        str(output_dir),
    ]

    if reference_path is not None:
        cmd.extend(["--reference", str(reference_path)])
    if stage1_only:
        cmd.append("--stage1-only")
    if headwear_mask:
        cmd.extend(["--headwear-mask", str(headwear_mask)])

    print("\n" + "=" * 70)
    print("  Executing Two-Stage Hairstyle Transfer Pipeline")
    print("=" * 70)
    print(f"  Target Image:     {target_path.name} ({target_path})")
    print(f"  Reference Hair:   {reference_path.name if reference_path else '(None - Stage 1 only)'}")
    print(f"  Stage 1 Only:     {stage1_only}")
    print(f"  Headwear Mask:    {headwear_mask or '(Auto detect)'}")
    print(f"  Stage 1 Steps:    {stage1_steps} (Bald Conversion)")
    if not stage1_only:
        print(f"  Stage 2 Steps:    {stage2_steps} (Reference Hair Conditioning & Edge Blending)")
        print(f"  IP-Adapter Scale: {scale}")
    print(f"  Random Seed:      {seed}")
    print(f"  Output Directory: {output_dir}")
    print("=" * 70 + "\n")

    result = subprocess.run(cmd)
    return result.returncode


def interactive_selection(assets_dir: Path = DEFAULT_ASSETS_DIR) -> Tuple[Optional[Path], Optional[Path]]:
    """Interactively display available assets and prompt the user to choose."""
    print_asset_list(assets_dir)
    print("Entering Interactive Selection Mode:")
    print("Type the filename, stem (e.g. 'target3'), or number of the image.\n")

    target_match: Optional[Path] = None
    while not target_match:
        try:
            val = input("  Select Target Portrait image: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
        if not val:
            continue
        target_match = match_image(val, assets_dir=assets_dir, role_hint="target")
        if not target_match:
            print(f"    [!] Could not find any target image matching '{val}'. Please try again.")

    print(f"  -> Selected Target: {target_match.name}")

    ref_match: Optional[Path] = None
    while not ref_match:
        try:
            val = input("  Select Hairstyle Reference image: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
        if not val:
            continue
        ref_match = match_image(val, assets_dir=assets_dir, role_hint="reference")
        if not ref_match:
            print(f"    [!] Could not find any reference image matching '{val}'. Please try again.")

    print(f"  -> Selected Reference: {ref_match.name}\n")
    return target_match, ref_match


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Two-Stage Hairstyle Transfer on any target and reference from assets/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default=None,
        help="Target portrait filename or stem (e.g. 'target3', 'target 2.jpg', 'target1.png')",
    )
    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        default=None,
        help="Hairstyle reference filename or stem (e.g. 'reference2', 'reference_4.jpg')",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available images in assets/ and exit",
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        default=str(DEFAULT_ASSETS_DIR),
        help="Path to assets folder",
    )
    parser.add_argument(
        "--stage1-steps",
        type=int,
        default=6,
        help="Stage 1 bald conversion diffusion inference steps",
    )
    parser.add_argument(
        "--stage2-steps",
        type=int,
        default=6,
        help="Stage 2 hairstyle transfer diffusion inference steps",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.75,
        help="IP-Adapter reference hairstyle conditioning scale",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation",
    )
    parser.add_argument(
        "--stage1-backend",
        type=str,
        choices=["auto", "sd-inpaint", "onnx"],
        default="auto",
        help="Stage 1 bald converter backend",
    )
    parser.add_argument(
        "--controlnet",
        type=str,
        choices=["auto", "on", "off"],
        default="auto",
        help="ControlNet spatial conditioning (auto-disables if weights not present)",
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Run Stage 1 bald conversion only and exit before Stage 2",
    )
    parser.add_argument(
        "--headwear-mask",
        type=str,
        default=None,
        help="Optional path to custom or manual headwear / stage 1 removal mask",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where experiment outputs are saved",
    )

    args = parser.parse_args()
    assets_dir = Path(args.assets_dir)

    # 1. Just list assets
    if args.list:
        print_asset_list(assets_dir)
        return 0

    # 2. Resolve or interactively select target and reference
    target_path: Optional[Path] = None
    ref_path: Optional[Path] = None

    if args.stage1_only:
        if args.target:
            target_path = match_image(args.target, assets_dir=assets_dir, role_hint="target")
            if not target_path:
                print(f"[ERROR] Could not resolve target image: '{args.target}' in {assets_dir.resolve()}", file=sys.stderr)
                print_asset_list(assets_dir)
                return 1
        else:
            print_asset_list(assets_dir)
            while not target_path:
                val = input("  Select Target Portrait image for Stage 1: ").strip()
                target_path = match_image(val, assets_dir=assets_dir, role_hint="target")
                if not target_path:
                    print(f"    [!] Could not find any target matching '{val}'. Please try again.")
    elif args.target and args.reference:
        target_path = match_image(args.target, assets_dir=assets_dir, role_hint="target")
        if not target_path:
            print(f"[ERROR] Could not resolve target image: '{args.target}' in {assets_dir.resolve()}", file=sys.stderr)
            print_asset_list(assets_dir)
            return 1

        ref_path = match_image(args.reference, assets_dir=assets_dir, role_hint="reference")
        if not ref_path:
            print(f"[ERROR] Could not resolve hairstyle reference image: '{args.reference}' in {assets_dir.resolve()}", file=sys.stderr)
            print_asset_list(assets_dir)
            return 1
    else:
        # If either is missing, trigger interactive selection
        target_path, ref_path = interactive_selection(assets_dir)

    if not target_path:
        print("[ERROR] Target portrait image must be selected.", file=sys.stderr)
        return 1

    # 3. Execute Two-Stage Pipeline
    return run_pipeline(
        target_path=target_path,
        reference_path=ref_path,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        scale=args.scale,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        stage1_backend=args.stage1_backend,
        controlnet=args.controlnet,
        stage1_only=args.stage1_only,
        headwear_mask=args.headwear_mask,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    sys.exit(main())
