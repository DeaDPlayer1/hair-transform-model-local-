"""
Unit and regression tests for Phase 3:
1. Selective reference hair mask hole filling.
2. Scale-independent similarity transform and NME validation.
3. Ear/sideburn boundary protection and continuity.
4. Aspect-ratio preserving Stage 2 diffusion preparation.
"""

import numpy as np
import cv2
import pytest
from PIL import Image

from src.hairstyle_alignment import (
    selective_fill_reference_mask_holes,
    compute_similarity_transform,
    build_facial_protection_mask,
    build_edge_repair_mask,
    composite_aligned_hair,
    ALIGNMENT_ANCHOR_INDICES,
    EAR_SEG_CLASSES,
)
from src.bald_converter import compute_inpaint_dimensions


class TestSelectiveHoleFilling:
    """Test Task 1: selective reference mask hole filling."""

    def test_tiny_enclosed_hole_filled(self):
        """Tiny enclosed holes (<100px) within hair mass should be filled."""
        h, w = 512, 512
        mask = np.zeros((h, w), dtype=np.uint8)
        # Create solid hair block
        mask[100:400, 100:400] = 255
        # Create tiny enclosed hole (area = 25 px)
        mask[200:205, 200:205] = 0

        rgb = np.full((h, w, 3), 180, dtype=np.uint8)  # studio backdrop
        rgb[100:400, 100:400] = 40  # dark hair
        rgb[200:205, 200:205] = 40  # dropped hair pixels inside hole

        filled, fill_mask, info = selective_fill_reference_mask_holes(mask, rgb)
        assert info["filled_count"] >= 1
        assert filled[202, 202] == 255
        assert fill_mask[202, 202] == 255

    def test_large_enclosed_region_preserved(self):
        """Large enclosed regions (>4000px) should NOT be filled indiscriminately."""
        h, w = 512, 512
        mask = np.full((h, w), 255, dtype=np.uint8)
        # Boundary is non-hair
        mask[:10, :] = 0
        mask[-10:, :] = 0
        mask[:, :10] = 0
        mask[:, -10:] = 0
        # Large central void (area 100x100 = 10000 px > max_thresh ~1048 px)
        mask[200:300, 200:300] = 0

        rgb = np.full((h, w, 3), 40, dtype=np.uint8)

        filled, fill_mask, info = selective_fill_reference_mask_holes(mask, rgb)
        # Large void must be preserved (not filled)
        assert filled[250, 250] == 0
        assert fill_mask[250, 250] == 0

    def test_external_curl_loop_preserved(self):
        """Curl loops showing studio backdrop color must be preserved."""
        h, w = 512, 512
        bg_color = np.array([180, 180, 180], dtype=np.uint8)  # Studio grey backdrop
        hair_color = np.array([45, 30, 20], dtype=np.uint8)

        rgb = np.full((h, w, 3), bg_color, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Hair region with an open curl loop of 300px showing background
        mask[100:400, 100:400] = 255
        rgb[100:400, 100:400] = hair_color

        # Hole in curl showing grey background
        mask[150:170, 150:165] = 0  # area = 300px
        rgb[150:170, 150:165] = bg_color  # matches corner background

        filled, fill_mask, info = selective_fill_reference_mask_holes(mask, rgb)
        # Loop matching background should be preserved
        assert filled[160, 155] == 0
        assert info["preserved_count"] >= 1

    def test_forehead_skin_pocket_preserved(self):
        """Gaps showing warm forehead skin inside hair fringe should be preserved as natural openings."""
        h, w = 512, 512
        bg_color = np.array([180, 180, 180], dtype=np.uint8)  # Studio grey
        hair_color = np.array([45, 30, 20], dtype=np.uint8)
        skin_color = np.array([210, 160, 135], dtype=np.uint8)  # Warm forehead skin

        rgb = np.full((h, w, 3), bg_color, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        mask[100:400, 100:400] = 255
        rgb[100:400, 100:400] = hair_color

        # Forehead skin gap (area = 300px)
        mask[220:240, 200:215] = 0
        rgb[220:240, 200:215] = skin_color

        filled, fill_mask, info = selective_fill_reference_mask_holes(mask, rgb)
        # Skin pocket should be preserved (not filled with donor skin)
        assert filled[230, 205] == 0
        assert fill_mask[230, 205] == 0
        assert info["preserved_count"] >= 1


class TestSimilarityTransform:
    """Test Task 2: scale-independent similarity transform and NME validation."""

    def test_scale_independent_ransac_high_res(self):
        """2048x2048 portrait with IOD ~500px should compute valid scale and low NME."""
        # Generate synthetic landmarks with known similarity transform: scale=1.8, rot=5 deg, tx=100, ty=200
        np.random.seed(42)
        src_pts = np.random.uniform(200, 800, size=(478, 2)).astype(np.float32)
        # Set synthetic eye landmarks for IOD
        src_pts[33] = [300.0, 400.0]
        src_pts[263] = [500.0, 400.0]

        theta = np.radians(5.0)
        s = 1.8
        M_gt = np.array([
            [s * np.cos(theta), -s * np.sin(theta), 100.0],
            [s * np.sin(theta),  s * np.cos(theta), 200.0],
        ], dtype=np.float32)

        dst_pts = (np.hstack([src_pts, np.ones((478, 1))]) @ M_gt.T).astype(np.float32)
        # Add small jitter (<10px, perfectly within 6% IOD on 2048px)
        dst_pts += np.random.normal(0, 3.0, dst_pts.shape).astype(np.float32)

        res = compute_similarity_transform(src_pts, dst_pts)
        assert res["valid"] is True
        assert res["confidence"] >= 0.80
        assert abs(res["scale"] - 1.8) < 0.1
        assert abs(res["rotation_deg"] - 5.0) < 2.0
        assert res["details"]["nme"] < 0.05

    def test_excessive_nme_rejection(self):
        """Scrambled/mismatched landmarks with high NME should be rejected."""
        np.random.seed(42)
        src_pts = np.random.uniform(200, 800, size=(478, 2)).astype(np.float32)
        # Completely random destination points (no valid affine fit)
        dst_pts = np.random.uniform(200, 800, size=(478, 2)).astype(np.float32)

        res = compute_similarity_transform(src_pts, dst_pts)
        assert res["valid"] is False


class TestEarSideburnBoundary:
    """Test Task 3: ear cartilage protection and boundary continuity."""

    def test_ear_cartilage_strictly_excluded_from_edge_repair_mask(self):
        """Actual ear cartilage must be strictly 0 in the edge repair mask."""
        h, w = 512, 512
        alpha = np.ones((h, w), dtype=np.float32)
        tgt_hair = np.zeros((h, w), dtype=np.uint8)

        # Create ear mask
        ear_mask = np.zeros((h, w), dtype=np.uint8)
        ear_mask[200:260, 100:140] = 255  # Left ear

        hard_prot = np.zeros((h, w), dtype=np.uint8)
        hard_prot[ear_mask > 0] = 255

        _, edge_np = build_edge_repair_mask(
            aligned_hair_alpha=alpha,
            target_hair_mask_np=tgt_hair,
            hard_protection=hard_prot,
            target_shape=(h, w),
            ear_mask_np=ear_mask,
            identity_protection=ear_mask,
        )

        # Ear cartilage must have exactly 0 in edge repair mask
        assert np.count_nonzero(edge_np[ear_mask > 0]) == 0

    def test_lower_face_protection_protects_cheeks(self):
        """Lower face polygon must protect cheek and jawline skin."""
        h, w = 512, 512
        # Synthetic landmarks with jaw and brows
        lms = np.zeros((478, 2), dtype=np.float32)
        # Eye landmarks for IOD
        lms[33] = [200.0, 200.0]
        lms[263] = [312.0, 200.0]
        # Jawline points along the bottom
        for i in range(17):
            lms[i] = [150.0 + i * 12.0, 300.0 + abs(i - 8) * 10.0]
        # Set brow points
        for i in range(105, 115):
            lms[i] = [150.0 + (i - 105) * 20.0, 180.0]

        target_landmarks = {"detected": True, "landmarks_2d": lms}
        prot = build_facial_protection_mask((h, w), target_landmarks, None, 15)

        assert prot.identity_protection is not None
        assert np.any(prot.identity_protection > 0)


class TestAspectRatioHandling:
    """Test Task 4: aspect-preserving diffusion preparation."""

    def test_square_image_dimensions(self):
        """Square images must return exact 512x512."""
        w, h = compute_inpaint_dimensions(512, 512, max_dim=512, min_dim=512)
        assert (w, h) == (512, 512)

        w2, h2 = compute_inpaint_dimensions(2048, 2048, max_dim=512, min_dim=512)
        assert (w2, h2) == (512, 512)

    def test_portrait_image_dimensions(self):
        """Portrait images must preserve aspect ratio and be divisible by 8."""
        w, h = compute_inpaint_dimensions(768, 1024, max_dim=512, min_dim=512)
        assert h == 512
        assert w == 384
        assert w % 8 == 0
        assert h % 8 == 0
        assert abs((w / h) - (768 / 1024)) < 0.01

    def test_landscape_image_dimensions(self):
        """Landscape images must preserve aspect ratio and be divisible by 8."""
        w, h = compute_inpaint_dimensions(1920, 1080, max_dim=512, min_dim=512)
        assert w == 512
        assert h == 288
        assert w % 8 == 0
        assert h % 8 == 0
        assert abs((w / h) - (1920 / 1080)) < 0.01
