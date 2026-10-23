"""
Tests for Natural & Photorealistic Hairstyle Transfer Enhancements:
- Edge repair mask confinement & validation (no outer curl inpainting, no empty-space bubbles)
- Separation of Stage 1 target hair mask vs. Stage 2 bald residual hair mask
- Photorealistic ambient contact shadow on forehead/temples
- Preservation of genuine hair strands during background decontamination
"""

import numpy as np
import pytest
from PIL import Image

from src.hair_mask_refinement import decontaminate_reference_hair_rgb
from src.hairstyle_alignment import (
    apply_contact_shadow,
    build_edge_repair_mask,
    composite_aligned_hair,
    validate_edge_repair_mask,
    create_repair_mask_validation_overlay,
    CompositeResult,
)


def test_edge_repair_mask_confines_to_permitted_interface_and_excludes_background():
    """
    Ensure edge_repair_mask does NOT inpaint outer curly silhouettes into empty background.
    """
    h, w = 256, 256
    # Donor hair mask covering top half including some background
    donor_hair_mask = np.zeros((h, w), dtype=np.uint8)
    donor_hair_mask[20:120, 40:216] = 255

    # Target segmentation: class 0 is background, class 1 is face/scalp skin
    seg_map = np.zeros((h, w), dtype=np.uint8)
    # Head ellipse in center
    yy, xx = np.ogrid[:h, :w]
    head_mask = ((yy - 128) ** 2 / (70 ** 2) + (xx - 128) ** 2 / (50 ** 2)) <= 1.0
    seg_map[head_mask] = 1

    # Empty residual hair on bald head
    residual_hair = np.zeros((h, w), dtype=np.uint8)

    _, repair_mask = build_edge_repair_mask(
        aligned_hair_alpha=(donor_hair_mask > 0).astype(np.float32),
        target_hair_mask_np=residual_hair,
        hard_protection=None,
        target_shape=(h, w),
        edge_width_px=10,
        target_segmentation_map=seg_map,
    )

    # Any pixel in background (seg_map == 0) MUST have 0 repair mask value
    background_repair_pixels = np.count_nonzero(repair_mask[seg_map == 0])
    assert background_repair_pixels == 0, f"Found {background_repair_pixels} repair pixels in background!"

    # Repair mask should only exist where donor hair intersects or touches head/scalp skin
    active_pixels = np.count_nonzero(repair_mask)
    assert active_pixels > 0, "Expected hairline repair pixels on the scalp interface"
    assert np.all(seg_map[repair_mask > 0] == 1), "All repair pixels must be on head/scalp skin"


def test_edge_repair_mask_ignores_empty_residual_hair():
    """
    When bald base has zero or minimal residual hair (< 500 px),
    build_edge_repair_mask should NOT create uncovered old hair inpainting regions.
    """
    h, w = 200, 200
    donor_hair = np.zeros((h, w), dtype=np.uint8)
    donor_hair[50:100, 50:150] = 255

    # Clean bald base (0 residual hair)
    clean_residual = np.zeros((h, w), dtype=np.uint8)

    seg_map = np.zeros((h, w), dtype=np.uint8)
    seg_map[40:160, 40:160] = 1  # head

    _, mask_clean = build_edge_repair_mask(
        aligned_hair_alpha=(donor_hair > 0).astype(np.float32),
        target_hair_mask_np=clean_residual,
        hard_protection=None,
        target_shape=(h, w),
        target_segmentation_map=seg_map,
    )

    # Now simulate a dirty bald base with 2000px residual hair outside donor coverage
    dirty_residual = np.zeros((h, w), dtype=np.uint8)
    dirty_residual[10:45, 50:150] = 255  # 35 * 100 = 3500 px of uncovered old hair

    _, mask_dirty = build_edge_repair_mask(
        aligned_hair_alpha=(donor_hair > 0).astype(np.float32),
        target_hair_mask_np=dirty_residual,
        hard_protection=None,
        target_shape=(h, w),
        target_segmentation_map=seg_map,
    )

    # The dirty mask must contain uncovered old hair repair pixels in y: 10..45
    assert np.count_nonzero(mask_dirty[10:45, :]) > 0
    # The clean mask must have ZERO repair pixels in that upper region
    assert np.count_nonzero(mask_clean[10:45, :]) == 0


def test_validate_edge_repair_mask():
    """
    Test validation metrics, fail-safe detection, and action decisions.
    """
    h, w = 200, 200
    empty_mask = np.zeros((h, w), dtype=np.uint8)
    alpha = np.ones((h, w), dtype=np.float32)
    res_empty = validate_edge_repair_mask(empty_mask, aligned_hair_alpha=alpha)
    assert res_empty["is_valid"] is True
    assert res_empty["active_pixels"] == 0
    assert res_empty["should_inpaint"] is False

    # Mask with identity intrusion
    intruding_mask = np.zeros((h, w), dtype=np.uint8)
    intruding_mask[90:110, 90:110] = 255
    hard_protection = np.zeros((h, w), dtype=np.uint8)
    hard_protection[100:150, 80:120] = 255  # overlaps 10x20 = 200px

    res_intruding = validate_edge_repair_mask(
        intruding_mask,
        aligned_hair_alpha=alpha,
        identity_protection=hard_protection,
    )
    assert res_intruding["is_valid"] is False
    assert res_intruding["identity_overlap_count"] == 200
    assert np.count_nonzero(res_intruding["sanitized_mask_np"][hard_protection > 0]) == 0

    # Mask with massive background coverage
    bg_mask = np.zeros((h, w), dtype=np.uint8)
    bg_mask[0:100, 0:100] = 255  # 10,000 px
    seg_map = np.zeros((h, w), dtype=np.uint8)
    seg_map[120:180, 50:150] = 1  # head is elsewhere

    res_bg = validate_edge_repair_mask(
        bg_mask,
        aligned_hair_alpha=alpha,
        target_segmentation_map=seg_map,
        max_outer_bg_ratio=0.05,
    )
    assert res_bg["is_valid"] is False
    assert res_bg["outer_bg_pixels"] == 10000


def test_contact_shadow_creation_and_bounds():
    """
    Test that contact shadow darkens skin beneath donor hair alpha edge
    and does NOT affect background or regions far from hair.
    """
    h, w = 100, 100
    base_rgb = np.full((h, w, 3), 200, dtype=np.uint8)  # uniform skin tone

    # Hair alpha overlapping skin from top to y=50
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[0:50, 20:80] = 1.0

    # Face skin mask covers y: 20..90, x: 20..80 (seg class 1)
    seg_map = np.zeros((h, w), dtype=np.uint8)
    seg_map[20:90, 20:80] = 1

    # Eye protection at y: 70..80
    hard_protection = np.zeros((h, w), dtype=np.uint8)
    hard_protection[70:80, 20:80] = 255

    shadowed_rgb, shadow_map = apply_contact_shadow(
        target_rgb=base_rgb,
        hair_alpha=alpha,
        target_segmentation_map=seg_map,
        identity_protection=hard_protection,
        shadow_strength=0.3,
    )

    # 1. Shadowed RGB values should be <= original RGB values (darkening)
    assert np.all(shadowed_rgb <= base_rgb)

    # 2. Shadow map must be 0 outside face_skin
    assert np.all(shadow_map[seg_map == 0] == 0)

    # 3. Shadow map must be 0 in protected eye region
    assert np.all(shadow_map[hard_protection > 0] == 0)

    # 4. There must be subtle darkening beneath terminating hair strands (e.g. y=50)
    contact_zone_darkening = 200 - int(shadowed_rgb[50, 50, 0])
    assert contact_zone_darkening > 0, "Contact shadow should darken skin immediately beneath hair strands"

    # 5. CompositeResult unpacking backwards compatibility
    donor_rgb = np.full((h, w, 3), 50, dtype=np.uint8)
    composite, eff_alpha = composite_aligned_hair(
        target_rgb=base_rgb,
        aligned_hair_rgb=donor_rgb,
        aligned_hair_alpha=alpha,
        hard_protection=hard_protection,
        contact_shadow_strength=0.25,
        target_segmentation_map=seg_map,
    )
    assert isinstance(composite, np.ndarray)
    assert isinstance(eff_alpha, np.ndarray)


def test_hair_decontamination_preserves_blonde_strands():
    """
    Ensure decontaminate_reference_hair_rgb does not strip legitimate light or blonde hair.
    """
    h, w = 100, 100
    # Clean warm blonde hair strand RGB (225, 195, 140)
    hair_rgb = np.full((h, w, 3), [225, 195, 140], dtype=np.uint8)
    hair_mask = np.full((h, w), 255, dtype=np.uint8)

    decon_rgb, unknown_region = decontaminate_reference_hair_rgb(
        reference_rgb=hair_rgb,
        ref_hair_binary=hair_mask,
        bounded_decontamination=True,
    )

    # Core blonde hair must be preserved, not zeroed out or contaminated
    assert np.mean(decon_rgb) > 100, "Blonde hair color should not be completely stripped"
