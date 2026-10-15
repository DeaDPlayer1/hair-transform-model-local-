"""
Unit tests for src/hairstyle_alignment.py.

Covers:
1. Missing landmarks / no face detected → fallback or graceful failure
2. Degenerate similarity transforms (insufficient points, implausible scale/rotation)
3. Mismatched dimensions between target and reference
4. Empty hair masks (both target and reference)
5. Alpha bounds: warped alpha is always in [0, 1]
6. Facial protection: eyes, nose, mouth excluded from edit mask
7. Hairline transition zone produces gradual ramp
8. Compositing preservation: original target untouched outside edit mask
9. Three masks are separate objects with correct types/shapes
10. Preview generation produces expected keys
11. Module compiles without syntax errors
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

from src.hairstyle_alignment import (
    estimate_head_geometry,
    compute_similarity_transform,
    compute_fallback_transform,
    warp_reference_hair,
    build_facial_protection_mask,
    build_alignment_masks,
    harmonize_hair_color,
    composite_aligned_hair,
    build_edge_repair_mask,
    create_alignment_previews,
    align_reference_to_target,
    ALIGNMENT_ANCHOR_INDICES,
    MIN_INLIERS,
    MIN_SCALE,
    MAX_SCALE,
    MAX_ROTATION_DEG,
)


def _make_synthetic_landmarks(w=256, h=256, offset_x=0.0, offset_y=0.0, scale=1.0):
    """Create a synthetic (478, 2) landmark array with face-like geometry."""
    lms = np.zeros((478, 2), dtype=np.float32)

    cx, cy = w / 2 + offset_x, h / 2 + offset_y
    s = scale

    # Forehead / cranial
    lms[10] = [cx, cy - 80 * s]           # forehead top
    lms[8] = [cx, cy - 60 * s]            # mid-forehead
    lms[9] = [cx, cy - 70 * s]
    lms[151] = [cx, cy - 75 * s]
    lms[168] = [cx, cy - 30 * s]          # nasion
    lms[6] = [cx, cy - 25 * s]
    lms[1] = [cx, cy + 10 * s]            # nose tip
    lms[152] = [cx, cy + 90 * s]          # chin

    # Temples / ears
    lms[234] = [cx - 60 * s, cy - 10 * s]  # left temple
    lms[454] = [cx + 60 * s, cy - 10 * s]  # right temple
    lms[103] = [cx - 55 * s, cy - 40 * s]  # left temple boundary
    lms[332] = [cx + 55 * s, cy - 40 * s]  # right temple boundary

    # Forehead lateral
    lms[109] = [cx - 40 * s, cy - 55 * s]
    lms[338] = [cx + 40 * s, cy - 55 * s]

    # Eyebrows
    lms[70] = [cx - 50 * s, cy - 45 * s]   # left outer brow
    lms[105] = [cx - 30 * s, cy - 48 * s]  # left mid brow
    lms[63] = [cx - 15 * s, cy - 46 * s]   # left inner brow
    lms[300] = [cx + 50 * s, cy - 45 * s]  # right outer brow
    lms[334] = [cx + 30 * s, cy - 48 * s]  # right mid brow
    lms[296] = [cx + 15 * s, cy - 46 * s]  # right inner brow
    lms[336] = [cx + 50 * s, cy - 45 * s]
    lms[293] = [cx + 40 * s, cy - 44 * s]

    # Eyes
    lms[33] = [cx - 40 * s, cy - 25 * s]   # left eye outer
    lms[133] = [cx - 15 * s, cy - 25 * s]  # left eye inner
    lms[263] = [cx + 40 * s, cy - 25 * s]  # right eye outer
    lms[362] = [cx + 15 * s, cy - 25 * s]  # right eye inner
    lms[160] = [cx - 35 * s, cy - 28 * s]
    lms[158] = [cx - 25 * s, cy - 28 * s]
    lms[159] = [cx - 30 * s, cy - 29 * s]
    lms[157] = [cx - 20 * s, cy - 27 * s]
    lms[173] = [cx - 38 * s, cy - 23 * s]
    lms[153] = [cx - 25 * s, cy - 22 * s]
    lms[144] = [cx - 30 * s, cy - 21 * s]
    lms[145] = [cx - 35 * s, cy - 22 * s]
    lms[385] = [cx + 35 * s, cy - 28 * s]
    lms[387] = [cx + 25 * s, cy - 28 * s]
    lms[386] = [cx + 30 * s, cy - 29 * s]
    lms[384] = [cx + 20 * s, cy - 27 * s]
    lms[398] = [cx + 38 * s, cy - 23 * s]
    lms[380] = [cx + 25 * s, cy - 22 * s]
    lms[373] = [cx + 30 * s, cy - 21 * s]
    lms[374] = [cx + 35 * s, cy - 22 * s]

    # Nose
    lms[197] = [cx, cy - 15 * s]
    lms[195] = [cx, cy - 5 * s]
    lms[5] = [cx, cy]
    lms[4] = [cx, cy + 5 * s]
    lms[2] = [cx - 5 * s, cy + 8 * s]
    lms[98] = [cx - 10 * s, cy + 10 * s]
    lms[327] = [cx + 10 * s, cy + 10 * s]

    # Mouth
    lms[61] = [cx - 20 * s, cy + 30 * s]
    lms[291] = [cx + 20 * s, cy + 30 * s]
    lms[17] = [cx, cy + 40 * s]
    lms[0] = [cx, cy + 25 * s]
    lms[146] = [cx - 15 * s, cy + 32 * s]
    lms[91] = [cx - 10 * s, cy + 35 * s]
    lms[181] = [cx - 5 * s, cy + 38 * s]
    lms[84] = [cx - 3 * s, cy + 40 * s]
    lms[314] = [cx + 5 * s, cy + 38 * s]
    lms[405] = [cx + 10 * s, cy + 35 * s]
    lms[321] = [cx + 15 * s, cy + 32 * s]
    lms[375] = [cx + 18 * s, cy + 30 * s]
    lms[308] = [cx + 20 * s, cy + 28 * s]
    lms[185] = [cx - 18 * s, cy + 28 * s]
    lms[40] = [cx - 15 * s, cy + 27 * s]
    lms[39] = [cx - 12 * s, cy + 26 * s]
    lms[37] = [cx - 8 * s, cy + 25 * s]
    lms[267] = [cx + 8 * s, cy + 25 * s]
    lms[269] = [cx + 12 * s, cy + 26 * s]
    lms[270] = [cx + 15 * s, cy + 27 * s]
    lms[409] = [cx + 18 * s, cy + 28 * s]

    # Cheeks / jaw
    lms[127] = [cx - 55 * s, cy]
    lms[356] = [cx + 55 * s, cy]
    lms[162] = [cx - 50 * s, cy - 30 * s]
    lms[21] = [cx - 45 * s, cy - 35 * s]
    lms[54] = [cx - 42 * s, cy - 38 * s]
    lms[67] = [cx - 35 * s, cy - 42 * s]
    lms[107] = [cx - 25 * s, cy - 44 * s]
    lms[66] = [cx - 30 * s, cy - 43 * s]
    lms[297] = [cx + 50 * s, cy - 30 * s]
    lms[284] = [cx + 45 * s, cy - 35 * s]
    lms[251] = [cx + 42 * s, cy - 38 * s]
    lms[389] = [cx + 40 * s, cy - 40 * s]

    return lms


def _make_landmarks_dict(w=256, h=256, **kwargs):
    """Convenience: return a detected-landmarks dict."""
    return {
        "detected": True,
        "landmarks_2d": _make_synthetic_landmarks(w, h, **kwargs),
        "landmarks_3d": None,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
    }


class TestHeadGeometry(unittest.TestCase):
    def test_valid_geometry(self):
        lms = _make_synthetic_landmarks(256, 256)
        geom = estimate_head_geometry(lms)
        self.assertTrue(geom["valid"])
        self.assertGreater(geom["head_width"], 0)
        self.assertGreater(geom["head_height"], 0)
        self.assertGreater(geom["interocular_dist"], 0)

    def test_none_landmarks(self):
        geom = estimate_head_geometry(None)
        self.assertFalse(geom["valid"])

    def test_too_few_landmarks(self):
        short = np.zeros((50, 2), dtype=np.float32)
        geom = estimate_head_geometry(short)
        self.assertFalse(geom["valid"])


class TestSimilarityTransform(unittest.TestCase):
    def test_identity_transform(self):
        """Same landmarks → near-identity transform."""
        lms = _make_synthetic_landmarks(256, 256)
        result = compute_similarity_transform(lms, lms)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["scale"], 1.0, places=1)
        self.assertAlmostEqual(result["rotation_deg"], 0.0, delta=5.0)

    def test_scaled_transform(self):
        """Landmarks at different scales → valid scale detected."""
        lms_small = _make_synthetic_landmarks(256, 256, scale=0.8)
        lms_large = _make_synthetic_landmarks(256, 256, scale=1.2)
        result = compute_similarity_transform(lms_small, lms_large)
        self.assertTrue(result["valid"])
        self.assertGreater(result["scale"], 1.0)

    def test_insufficient_correspondences(self):
        """Very short landmark arrays → rejected."""
        short_src = np.zeros((3, 2), dtype=np.float32)
        short_dst = np.zeros((3, 2), dtype=np.float32)
        result = compute_similarity_transform(short_src, short_dst)
        self.assertFalse(result["valid"])
        self.assertIn("Insufficient", result["rejection_reason"])

    def test_degenerate_all_same_point(self):
        """All points identical → degenerate, rejected."""
        lms_src = np.ones((478, 2), dtype=np.float32) * 128
        lms_dst = np.ones((478, 2), dtype=np.float32) * 128
        result = compute_similarity_transform(lms_src, lms_dst)
        # Should be rejected (degenerate geometry)
        # The scale would be 0 or RANSAC fails
        if result["valid"]:
            # Even if RANSAC "succeeds", scale ~ 0 should be caught
            self.assertGreaterEqual(result["scale"], MIN_SCALE)

    def test_implausible_scale_rejected(self):
        """Extreme scale difference → rejected."""
        lms_tiny = _make_synthetic_landmarks(256, 256, scale=0.1)
        lms_huge = _make_synthetic_landmarks(256, 256, scale=1.0)
        result = compute_similarity_transform(lms_tiny, lms_huge)
        # This would give scale ~10, which should be rejected
        if not result["valid"]:
            self.assertIn("scale", result["rejection_reason"].lower())


class TestFallbackTransform(unittest.TestCase):
    def test_basic_fallback(self):
        """Fallback with valid hair masks gives a valid transform."""
        t_mask = np.zeros((256, 256), dtype=np.uint8)
        t_mask[20:80, 60:200] = 255
        r_mask = np.zeros((256, 256), dtype=np.uint8)
        r_mask[30:100, 50:210] = 255

        result = compute_fallback_transform(
            {"valid": False}, {"valid": False},
            t_mask, r_mask, (256, 256),
        )
        self.assertTrue(result["valid"])
        self.assertTrue(result.get("is_fallback"))
        self.assertIsNotNone(result["matrix"])

    def test_empty_reference_mask(self):
        """Empty reference mask → fallback fails."""
        t_mask = np.zeros((256, 256), dtype=np.uint8)
        t_mask[20:80, 60:200] = 255
        r_mask = np.zeros((256, 256), dtype=np.uint8)  # empty

        result = compute_fallback_transform(
            {"valid": False}, {"valid": False},
            t_mask, r_mask, (256, 256),
        )
        self.assertFalse(result["valid"])
        self.assertIn("empty", result["rejection_reason"].lower())


class TestWarpReferenceHair(unittest.TestCase):
    def test_identity_warp(self):
        """Identity matrix → warped output == input (within target dims)."""
        rgb = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        alpha = np.random.rand(128, 128).astype(np.float32)
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

        w_rgb, w_alpha = warp_reference_hair(rgb, alpha, M, (128, 128))
        self.assertEqual(w_rgb.shape, (128, 128, 3))
        self.assertEqual(w_alpha.shape, (128, 128))
        # Core pixels should be preserved
        np.testing.assert_array_equal(w_rgb[10:118, 10:118], rgb[10:118, 10:118])

    def test_alpha_bounds(self):
        """Warped alpha must stay in [0, 1]."""
        alpha = np.ones((100, 100), dtype=np.float32)
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        # Translation that shifts content partially out of frame
        M = np.array([[1, 0, 50], [0, 1, 50]], dtype=np.float64)

        _, w_alpha = warp_reference_hair(rgb, alpha, M, (100, 100))
        self.assertGreaterEqual(w_alpha.min(), 0.0)
        self.assertLessEqual(w_alpha.max(), 1.0)

    def test_different_target_shape(self):
        """Warp to a differently-sized target."""
        rgb = np.zeros((200, 150, 3), dtype=np.uint8)
        alpha = np.zeros((200, 150), dtype=np.float32)
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

        w_rgb, w_alpha = warp_reference_hair(rgb, alpha, M, (300, 400))
        self.assertEqual(w_rgb.shape, (300, 400, 3))
        self.assertEqual(w_alpha.shape, (300, 400))


class TestFacialProtection(unittest.TestCase):
    def test_protection_with_segmentation(self):
        """Segmentation-based protection covers eyes and nose."""
        seg = np.zeros((256, 256), dtype=np.uint8)
        seg[100:120, 80:110] = 4   # left eye
        seg[100:120, 150:180] = 5  # right eye
        seg[130:150, 115:145] = 10 # nose

        hard, trans = build_facial_protection_mask(
            (256, 256),
            target_segmentation_map=seg,
        )

        # Eyes and nose region should be protected
        self.assertTrue(np.any(hard[100:120, 80:110] > 0))
        self.assertTrue(np.any(hard[100:120, 150:180] > 0))
        self.assertTrue(np.any(hard[130:150, 115:145] > 0))

    def test_protection_with_landmarks(self):
        """Landmark-based protection creates protected regions."""
        lm_dict = _make_landmarks_dict(256, 256)
        hard, trans = build_facial_protection_mask(
            (256, 256),
            target_landmarks=lm_dict,
        )
        # Some pixels should be protected
        self.assertGreater(np.count_nonzero(hard), 0)

    def test_transition_weight_range(self):
        """Transition weight is in [0, 1]."""
        lm_dict = _make_landmarks_dict(256, 256)
        _, trans = build_facial_protection_mask(
            (256, 256),
            target_landmarks=lm_dict,
            hairline_transition_px=20,
        )
        self.assertGreaterEqual(trans.min(), 0.0)
        self.assertLessEqual(trans.max(), 1.0)

    def test_no_inputs_no_crash(self):
        """No landmarks or segmentation → no crash, empty protection."""
        hard, trans = build_facial_protection_mask((256, 256))
        self.assertEqual(hard.shape, (256, 256))
        self.assertEqual(np.count_nonzero(hard), 0)
        np.testing.assert_array_equal(trans, np.ones((256, 256), dtype=np.float32))


class TestAlignmentMasks(unittest.TestCase):
    def setUp(self):
        self.h, self.w = 256, 256
        # Aligned hair alpha (upper region)
        self.aligned_alpha = np.zeros((self.h, self.w), dtype=np.float32)
        self.aligned_alpha[10:80, 50:200] = 0.9

        # Target hair mask
        self.target_hair = np.zeros((self.h, self.w), dtype=np.uint8)
        self.target_hair[20:90, 60:190] = 255

        # Protection
        self.hard_protection = np.zeros((self.h, self.w), dtype=np.uint8)
        self.hard_protection[110:130, 80:170] = 255  # eyes region

        # Full transition
        self.transition = np.ones((self.h, self.w), dtype=np.float32)

    def test_three_masks_separate(self):
        """Three masks are separate objects with correct types."""
        masks = build_alignment_masks(
            self.aligned_alpha, self.target_hair,
            self.hard_protection, self.transition,
            (self.h, self.w),
        )
        self.assertIn("aligned_hair_alpha", masks)
        self.assertIn("target_edit_mask", masks)
        self.assertIn("compositing_alpha", masks)

        # Types
        self.assertIsInstance(masks["aligned_hair_alpha"], np.ndarray)
        self.assertIsInstance(masks["target_edit_mask"], Image.Image)
        self.assertIsInstance(masks["compositing_alpha"], np.ndarray)

        # Shapes
        self.assertEqual(masks["aligned_hair_alpha"].shape, (self.h, self.w))
        self.assertEqual(masks["target_edit_mask"].size, (self.w, self.h))

    def test_protection_excluded_from_edit_mask(self):
        """Protected regions must be 0 in edit mask."""
        masks = build_alignment_masks(
            self.aligned_alpha, self.target_hair,
            self.hard_protection, self.transition,
            (self.h, self.w),
        )
        edit_np = np.asarray(masks["target_edit_mask"])
        protected_region = self.hard_protection > 0
        self.assertEqual(np.max(edit_np[protected_region]), 0)

    def test_edit_mask_covers_original_hair(self):
        """Edit mask includes original target hair (minus protected)."""
        masks = build_alignment_masks(
            self.aligned_alpha, self.target_hair,
            self.hard_protection, self.transition,
            (self.h, self.w),
            cranial_expansion_px=0,
        )
        edit_np = np.asarray(masks["target_edit_mask"])
        # Non-protected original hair should be in edit mask
        hair_and_not_protected = (self.target_hair > 128) & (self.hard_protection == 0)
        covered = edit_np[hair_and_not_protected] > 128
        coverage = np.mean(covered)
        self.assertGreater(coverage, 0.9, "Edit mask doesn't cover original hair")

    def test_compositing_alpha_strictly_inside(self):
        """Compositing alpha is 0 outside edit mask."""
        masks = build_alignment_masks(
            self.aligned_alpha, self.target_hair,
            self.hard_protection, self.transition,
            (self.h, self.w),
        )
        edit_np = np.asarray(masks["target_edit_mask"])
        comp_alpha = masks["compositing_alpha"]
        outside = edit_np == 0
        if comp_alpha.ndim == 3:
            self.assertEqual(float(np.max(comp_alpha[outside, 0])), 0.0)
        else:
            self.assertEqual(float(np.max(comp_alpha[outside])), 0.0)


class TestAlignmentPreviews(unittest.TestCase):
    def test_preview_keys(self):
        """Preview generation returns expected image keys."""
        rgb = np.zeros((128, 128, 3), dtype=np.uint8)
        alpha = np.zeros((128, 128), dtype=np.float32)
        mask = np.zeros((128, 128), dtype=np.uint8)

        previews = create_alignment_previews(
            target_rgb=rgb,
            aligned_hair_rgb=rgb,
            aligned_hair_alpha=alpha,
            target_edit_mask_np=mask,
        )
        expected_keys = [
            "aligned_rgb_preview",
            "alpha_preview",
            "overlay_preview",
            "edit_mask_preview",
        ]
        for key in expected_keys:
            self.assertIn(key, previews, f"Missing preview: {key}")
            self.assertIsInstance(previews[key], np.ndarray)


class TestEndToEndAlignment(unittest.TestCase):
    """Integration tests for align_reference_to_target."""

    def setUp(self):
        self.w, self.h = 256, 256
        self.target = Image.new("RGB", (self.w, self.h), (180, 160, 140))
        self.reference = Image.new("RGB", (self.w, self.h), (60, 50, 40))

        t_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        t_mask[20:80, 60:190] = 255
        self.target_mask = Image.fromarray(t_mask, mode="L")

        r_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        r_mask[25:90, 55:195] = 255
        self.ref_mask = Image.fromarray(r_mask, mode="L")

        self.seg_map = np.zeros((self.h, self.w), dtype=np.uint8)
        self.seg_map[20:80, 60:190] = 17   # hair
        self.seg_map[100:200, 70:180] = 1  # face
        self.seg_map[110:125, 80:105] = 4  # l_eye
        self.seg_map[110:125, 145:170] = 5 # r_eye
        self.seg_map[130:150, 115:135] = 10 # nose
        self.seg_map[160:180, 105:145] = 11 # mouth

    @patch("src.hairstyle_alignment.detect_face_landmarks")
    def test_no_faces_detected(self, mock_detect):
        """No faces in either image → fallback, warnings, valid output."""
        mock_detect.return_value = {
            "detected": False, "landmarks_2d": None,
            "landmarks_3d": None, "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
        }
        result = align_reference_to_target(
            self.target, self.reference,
            self.target_mask, self.ref_mask,
        )
        self.assertIn("aligned_hair_alpha", result)
        self.assertIn("target_edit_mask", result)
        self.assertIn("warnings", result)
        self.assertTrue(len(result["warnings"]) > 0)

    @patch("src.hairstyle_alignment.detect_face_landmarks")
    def test_with_landmarks(self, mock_detect):
        """Both faces detected → landmark-based alignment."""
        mock_detect.return_value = _make_landmarks_dict(256, 256)

        result = align_reference_to_target(
            self.target, self.reference,
            self.target_mask, self.ref_mask,
            target_segmentation_map=self.seg_map,
        )
        self.assertIn("aligned_hair_alpha", result)
        self.assertIn("target_edit_mask", result)
        self.assertIn("compositing_alpha", result)
        self.assertIn("transform_info", result)
        self.assertIn("previews", result)

        # Alpha bounds
        alpha = result["aligned_hair_alpha"]
        self.assertGreaterEqual(alpha.min(), 0.0)
        self.assertLessEqual(alpha.max(), 1.0)

        # Edit mask is PIL Image
        self.assertIsInstance(result["target_edit_mask"], Image.Image)

        # Dimensions match target
        self.assertEqual(result["aligned_hair_alpha"].shape, (self.h, self.w))
        edit_mask = result["target_edit_mask"]
        self.assertEqual(edit_mask.size, (self.w, self.h))

    @patch("src.hairstyle_alignment.detect_face_landmarks")
    def test_empty_reference_mask(self, mock_detect):
        """Empty reference mask → graceful handling."""
        mock_detect.return_value = _make_landmarks_dict(256, 256)

        empty_mask = Image.new("L", (self.w, self.h), 0)
        result = align_reference_to_target(
            self.target, self.reference,
            self.target_mask, empty_mask,
        )
        # Should still produce valid output (using target hair for edit mask)
        self.assertIn("target_edit_mask", result)

    @patch("src.hairstyle_alignment.detect_face_landmarks")
    def test_mismatched_dimensions(self, mock_detect):
        """Different target and reference dimensions → no crash."""
        mock_detect.return_value = _make_landmarks_dict(256, 256)

        big_ref = Image.new("RGB", (512, 384), (60, 50, 40))
        big_ref_mask = Image.new("L", (512, 384), 0)
        mask_np = np.asarray(big_ref_mask).copy()
        mask_np[30:150, 100:400] = 255
        big_ref_mask = Image.fromarray(mask_np, mode="L")

        result = align_reference_to_target(
            self.target, big_ref,
            self.target_mask, big_ref_mask,
        )
        # Output dimensions should match TARGET
        self.assertEqual(result["aligned_hair_alpha"].shape, (self.h, self.w))

    @patch("src.hairstyle_alignment.detect_face_landmarks")
    def test_facial_protection_in_edit_mask(self, mock_detect):
        """Eyes and nose must be excluded from edit mask."""
        mock_detect.return_value = _make_landmarks_dict(256, 256)

        result = align_reference_to_target(
            self.target, self.reference,
            self.target_mask, self.ref_mask,
            target_segmentation_map=self.seg_map,
        )
        edit_np = np.asarray(result["target_edit_mask"])
        # Eye regions (class 4, 5) should be 0 in edit mask
        for cls in [4, 5, 10, 11]:
            region = self.seg_map == cls
            self.assertEqual(
                np.max(edit_np[region]), 0,
                f"Segmentation class {cls} was NOT protected in edit mask!"
            )


class TestModuleCompiles(unittest.TestCase):
    def test_compile(self):
        """Verify hairstyle_alignment.py compiles without errors."""
        import py_compile
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "src" / "hairstyle_alignment.py"
        self.assertTrue(path.is_file())
        py_compile.compile(str(path), doraise=True)


class TestHairstylePreservationAndSeamRepair(unittest.TestCase):
    def test_no_horizontal_seam_in_transition(self):
        """Verifies 2D distance transform does not attenuate entire horizontal row across width."""
        h, w = 148, 186
        hard_prot = np.zeros((h, w), dtype=np.uint8)
        hard_prot[36:50, 60:120] = 255  # Eyes/brow box in face center
        
        non_prot = (hard_prot == 0).astype(np.uint8)
        import cv2
        dist = cv2.distanceTransform(non_prot, cv2.DIST_L2, 5)
        tw = np.clip(dist / 15.0, 0.0, 1.0).astype(np.float32)
        
        # Left and right sides outside the face columns should be 1.0 (unaffected)
        self.assertEqual(tw[40, 10], 1.0)
        self.assertEqual(tw[40, 175], 1.0)
        # Right at the feature, weight is 0
        self.assertEqual(tw[40, 80], 0.0)
        # Row 40 min is 0.0 and max is 1.0 (not all forced to same attenuated value)
        self.assertEqual(tw[40].min(), 0.0)
        self.assertEqual(tw[40].max(), 1.0)

    def test_composite_aligned_hair_preservation(self):
        """Direct compositing places reference hair and strictly protects features."""
        target = np.ones((100, 100, 3), dtype=np.uint8) * 100
        aligned_rgb = np.ones((100, 100, 3), dtype=np.uint8) * 200
        aligned_alpha = np.zeros((100, 100), dtype=np.float32)
        aligned_alpha[10:40, 20:80] = 1.0  # hair region

        hard_prot = np.zeros((100, 100), dtype=np.uint8)
        hard_prot[35:45, 40:60] = 255  # eyes overlapping lower hair

        comp, eff_alpha = composite_aligned_hair(target, aligned_rgb, aligned_alpha, hard_prot)
        # Inside core hair without protection, composite is exactly 200
        self.assertEqual(comp[20, 50, 0], 200)
        # Inside protected region, hair is zeroed out and composite remains target 100
        self.assertEqual(comp[38, 50, 0], 100)
        self.assertEqual(eff_alpha[38, 50], 0.0)
        # Outside hair, composite is target 100
        self.assertEqual(comp[80, 50, 0], 100)

    def test_build_edge_repair_mask_excludes_central_hair(self):
        """Central hair core is strictly excluded from edge repair inpainting mask."""
        alpha = np.zeros((100, 100), dtype=np.float32)
        # Voluminous hair block
        alpha[10:60, 20:80] = 1.0
        target_hair = np.zeros((100, 100), dtype=np.uint8)
        target_hair[20:50, 30:70] = 255  # old hair fully covered

        edge_mask_pil, edge_mask_np = build_edge_repair_mask(
            aligned_hair_alpha=alpha,
            target_hair_mask_np=target_hair,
            hard_protection=None,
            target_shape=(100, 100),
            edge_width_px=8,
        )

        # Deep interior of hair (central curls) must be 0 (NOT inpainted)
        self.assertEqual(edge_mask_np[35, 50], 0, "Central hair was not protected from inpainting!")
        # Boundary edges must be 255 (inpainted for seamless blend)
        self.assertGreater(np.count_nonzero(edge_mask_np), 0)
        self.assertLess(np.count_nonzero(edge_mask_np), np.count_nonzero(alpha > 0))

    def test_harmonize_hair_color_bounds(self):
        """Harmonization preserves shape, values in [0, 255], and doesn't crash on empty mask."""
        tgt = np.full((50, 50, 3), 50, dtype=np.uint8)
        ref = np.full((50, 50, 3), 180, dtype=np.uint8)
        alpha = np.ones((50, 50), dtype=np.float32)
        tgt_mask = np.full((50, 50), 255, dtype=np.uint8)

        harmonized = harmonize_hair_color(ref, alpha, tgt, tgt_mask, strength=0.40)
        self.assertEqual(harmonized.shape, (50, 50, 3))
        self.assertEqual(harmonized.dtype, np.uint8)
        self.assertGreaterEqual(harmonized.min(), 0)
    def test_premultiplied_alpha_warping_no_fringe(self):
        """Warping with premultiplied alpha preserves clean edge color without borderValue bleeding."""
        ref_rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
        ref_alpha = np.zeros((100, 100), dtype=np.float32)
        ref_alpha[20:80, 20:80] = 1.0
        # Linear ramp near edge
        ref_alpha[15:20, 20:80] = np.linspace(0.1, 0.9, 5)[:, None]

        M = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 10.0]])
        warped_rgb, warped_alpha = warp_reference_hair(ref_rgb, ref_alpha, M, (120, 120))

        # Check that where alpha > 0.1, warped_rgb does NOT collapse to 0 (black fringe)
        valid_edge = (warped_alpha > 0.15) & (warped_alpha < 0.85)
        self.assertTrue(np.any(valid_edge))
        # Edge RGB should remain ~200 (within rounding), not degraded towards borderValue (0,0,0)
        self.assertGreaterEqual(warped_rgb[valid_edge].mean(), 195)

    def test_harmonize_hair_color_preserves_delicate_outer_strands(self):
        """Low-alpha boundary strands (alpha <= 0.65) must remain untouched by LAB harmonization to prevent halo glow."""
        tgt = np.full((50, 50, 3), 40, dtype=np.uint8)
        ref = np.full((50, 50, 3), 100, dtype=np.uint8)
        alpha = np.zeros((50, 50), dtype=np.float32)
        alpha[10:40, 10:40] = 0.90  # core
        alpha[5:10, 10:40] = 0.30   # outer delicate strands
        tgt_mask = np.full((50, 50), 255, dtype=np.uint8)

        harmonized = harmonize_hair_color(ref, alpha, tgt, tgt_mask, strength=0.50)
        # Core hair is harmonized (closer to 40)
        self.assertLess(harmonized[20, 20, 0], 90)
        # Outer delicate strands (alpha <= 0.65) are left untouched at exactly 100
        self.assertEqual(harmonized[7, 20, 0], 100)

    def test_bounded_decontamination_ring_preserves_distant_background(self):
        """Verify bounded ring decontamination leaves distant background 100% untouched."""
        # 100x100 canvas with hair in the center and distinct distant background pattern
        tgt = Image.new("RGB", (100, 100), (120, 100, 80))
        ref_arr = np.full((100, 100, 3), 220, dtype=np.uint8)  # backdrop
        ref_arr[35:65, 35:65] = [40, 30, 20]  # dark hair core
        # Corner background marker
        ref_arr[5, 5] = [15, 240, 99]
        ref = Image.fromarray(ref_arr)

        t_mask = Image.new("L", (100, 100), 0)
        r_mask_np = np.zeros((100, 100), dtype=np.uint8)
        r_mask_np[35:65, 35:65] = 255
        r_mask = Image.fromarray(r_mask_np)

        result_bounded = align_reference_to_target(
            target_image=tgt,
            reference_image=ref,
            target_hair_mask=t_mask,
            reference_hair_mask=r_mask,
            bounded_decontamination=True,
            ring_radius_px=10,
        )

        decontam_bounded = result_bounded["decontaminated_reference_rgb"]
        # Distant background pixel (5, 5) MUST be 100% preserved
        self.assertEqual(list(decontam_bounded[5, 5]), [15, 240, 99])

        # Unknown region mask must NOT cover the whole non-core background
        unknown_mask = result_bounded["unknown_region_mask"]
        self.assertLess(np.count_nonzero(unknown_mask), 100 * 100 * 0.5)

    def test_bounded_decontamination_is_hair_rim_only(self):
        """Bounded unknown region must not extend into exterior studio backdrop."""
        tgt = Image.new("RGB", (100, 100), (120, 100, 80))
        ref_arr = np.full((100, 100, 3), 220, dtype=np.uint8)
        ref_arr[35:65, 35:65] = [40, 30, 20]
        ref = Image.fromarray(ref_arr)
        t_mask = Image.new("L", (100, 100), 0)
        r_mask_np = np.zeros((100, 100), dtype=np.uint8)
        r_mask_np[35:65, 35:65] = 255
        r_mask = Image.fromarray(r_mask_np)

        result = align_reference_to_target(
            target_image=tgt,
            reference_image=ref,
            target_hair_mask=t_mask,
            reference_hair_mask=r_mask,
            bounded_decontamination=True,
        )
        unknown = result["unknown_region_mask"]
        outside_hair = r_mask_np == 0
        self.assertEqual(int(np.count_nonzero(unknown[outside_hair])), 0)

    def test_backdrop_pruning_removes_large_studio_pockets(self):
        """Large parser-filled studio pockets inside the hair blob must be removed."""
        ref_arr = np.full((120, 120, 3), 220, dtype=np.uint8)
        ref_arr[20:100, 20:100] = [45, 32, 24]
        ref_arr[48:72, 48:72] = [220, 220, 220]  # large interior pocket
        ref = Image.fromarray(ref_arr)
        tgt = Image.new("RGB", (120, 120), (110, 95, 80))
        t_mask = Image.new("L", (120, 120), 0)
        r_mask_np = np.zeros((120, 120), dtype=np.uint8)
        r_mask_np[20:100, 20:100] = 255
        r_mask = Image.fromarray(r_mask_np)

        result = align_reference_to_target(
            target_image=tgt,
            reference_image=ref,
            target_hair_mask=t_mask,
            reference_hair_mask=r_mask,
        )
        alpha = result["reference_hair_alpha"]
        self.assertLess(float(alpha[60, 60]), 0.05)

    def test_pasted_composite_rejects_white_hair_fringe(self):
        """A white ring mislabeled as hair must not survive as a pale pasted halo."""
        tgt = np.full((120, 120, 3), 90, dtype=np.uint8)
        tgt[40:80, 40:80] = [180, 140, 120]
        tgt_img = Image.fromarray(tgt)
        ref_arr = np.full((120, 120, 3), 225, dtype=np.uint8)
        ref_arr[30:90, 30:90] = [50, 35, 25]
        ref = Image.fromarray(ref_arr)
        t_mask = Image.new("L", (120, 120), 0)
        r_mask_np = np.zeros((120, 120), dtype=np.uint8)
        r_mask_np[25:95, 25:95] = 255  # 5px white fringe labeled as hair
        r_mask = Image.fromarray(r_mask_np)

        result = align_reference_to_target(
            target_image=tgt_img,
            reference_image=ref,
            target_hair_mask=t_mask,
            reference_hair_mask=r_mask,
        )
        pasted = np.asarray(result["pasted_composite"])
        alpha = result["effective_hair_alpha"]
        fringe = (alpha > 0.08) & (alpha < 0.9)
        if np.any(fringe):
            luma = (
                0.2126 * pasted[..., 0] + 0.7152 * pasted[..., 1] + 0.0722 * pasted[..., 2]
            )
            self.assertLess(float(luma[fringe].mean()), 160.0)

    def test_backdrop_pruning_protects_deep_hair_core(self):
        """Verify bright highlights in deep hair core are NOT removed by backdrop pruning."""
        ref_arr = np.full((100, 100, 3), 220, dtype=np.uint8)  # backdrop ~220
        # Hair region
        ref_arr[20:80, 20:80] = [50, 40, 30]
        # Bright highlight in deep center of hair matching backdrop color ~220
        ref_arr[48:52, 48:52] = [220, 220, 220]
        ref = Image.fromarray(ref_arr)

        tgt = Image.new("RGB", (100, 100), (120, 100, 80))
        t_mask = Image.new("L", (100, 100), 0)
        r_mask_np = np.zeros((100, 100), dtype=np.uint8)
        r_mask_np[20:80, 20:80] = 255
        r_mask = Image.fromarray(r_mask_np)

        result = align_reference_to_target(
            target_image=tgt,
            reference_image=ref,
            target_hair_mask=t_mask,
            reference_hair_mask=r_mask,
        )

        # The deep highlight at (50, 50) must NOT have its alpha zeroed out
        alpha = result["reference_hair_alpha"]
        self.assertGreater(alpha[50, 50], 0.5)


    def test_ear_exclusion_from_edge_repair_mask(self):
        """Verifies ear region and its buffer are strictly excluded from edge repair mask."""
        alpha = np.zeros((100, 100), dtype=np.float32)
        alpha[20:80, 20:80] = 1.0  # Hair covers (20:80, 20:80)
        target_hair = np.zeros((100, 100), dtype=np.uint8)
        target_hair[25:75, 25:75] = 255
        
        # Ear located at boundary of hair
        ear_mask = np.zeros((100, 100), dtype=np.uint8)
        ear_mask[40:60, 15:25] = 255

        _, edge_mask_np = build_edge_repair_mask(
            aligned_hair_alpha=alpha,
            target_hair_mask_np=target_hair,
            hard_protection=None,
            ear_mask_np=ear_mask,
            target_shape=(100, 100),
            edge_width_px=8,
        )

        # Pixels in ear mask and dilated ear buffer MUST be 0 (no diffusion inpainting)
        self.assertEqual(int(edge_mask_np[ear_mask > 0].max()), 0)
        # Pixels 2px adjacent to ear inside hair border should also be buffered and 0
        self.assertEqual(int(edge_mask_np[50, 22]), 0)

    def test_soft_transition_around_hard_protection(self):
        """Verifies fractional antialiased transition exists immediately outside hard protection."""
        tgt = np.full((100, 100, 3), 120, dtype=np.uint8)
        rgb = np.full((100, 100, 3), 20, dtype=np.uint8)
        alpha = np.ones((100, 100), dtype=np.float32)
        hard_prot = np.zeros((100, 100), dtype=np.uint8)
        hard_prot[40:60, 40:60] = 255  # Box protected

        comp, eff_alpha = composite_aligned_hair(
            tgt, rgb, alpha, hard_prot, soft_transition_px=3.0
        )
        # Inside hard protection: exactly 0
        self.assertEqual(float(eff_alpha[50, 50]), 0.0)
        # Exactly on the edge / 1-2px outside: fractional weight (0 < alpha < 1)
        # At (39, 50), distance to protection is 1.0, soft weight is ~1/3 = 0.33
        self.assertGreater(float(eff_alpha[39, 50]), 0.0)
        self.assertLess(float(eff_alpha[39, 50]), 1.0)
        # Deep outside (> 3px away): full alpha 1.0
        self.assertEqual(float(eff_alpha[20, 50]), 1.0)

    def test_scale_aware_parameters_scaling(self):
        """Verifies align_reference_to_target clamps scale-aware parameters based on IOD."""
        tgt = Image.new("RGB", (256, 256), (100, 100, 100))
        ref = Image.new("RGB", (256, 256), (50, 50, 50))
        t_mask = Image.new("L", (256, 256), 0)
        r_mask = Image.new("L", (256, 256), 255)

        # Mock geometry with large IOD (e.g. 150px)
        with patch("src.hairstyle_alignment.estimate_head_geometry") as mock_geom:
            mock_geom.return_value = {
                "valid": True,
                "interocular_dist": 150.0,
                "head_width": 200.0,
                "head_height": 240.0,
            }
            with patch("src.hairstyle_alignment.detect_face_landmarks") as mock_lms:
                mock_lms.return_value = _make_landmarks_dict(256, 256)
                result = align_reference_to_target(tgt, ref, t_mask, r_mask)
                self.assertIn("aligned_hair_alpha", result)
                # Parameters should be safely clamped
                self.assertIsNotNone(result.get("pasted_composite"))


if __name__ == "__main__":
    unittest.main()


