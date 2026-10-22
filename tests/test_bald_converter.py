"""
Comprehensive Unit & Regression Tests for Stage 1 Bald Converter.
Fast, deterministic tests using mocked diffusion pipeline.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np
from PIL import Image

from src.bald_converter import (
    BaldConverterConfig,
    BaldConversionResult,
    SDInpaintBaldConverter,
    ONNXBaldConverter,
    create_bald_converter,
    compute_face_geometry,
    compute_inpaint_dimensions,
    evaluate_stage1_quality_gate,
)
from src.headwear_detection import HeadwearDetectionResult


class TestBaldConverterDynamic(unittest.TestCase):
    def setUp(self):
        # Base synthetic image 200x200
        self.w, self.h = 200, 200
        self.target_rgb = Image.new("RGB", (self.w, self.h), color=(220, 200, 180))
        self.target_rgba = Image.new("RGBA", (self.w, self.h), color=(220, 200, 180, 255))

        # Synthetic hair mask: upper head (0 to 80y, 40 to 160x)
        hair_arr = np.zeros((self.h, self.w), dtype=np.uint8)
        hair_arr[:80, 40:160] = 255
        self.hair_mask = Image.fromarray(hair_arr, mode="L")

        # Synthetic headwear mask
        hw_arr = np.zeros((self.h, self.w), dtype=np.uint8)
        hw_arr[:50, 30:170] = 255
        self.headwear_mask = Image.fromarray(hw_arr, mode="L")

        # Segmentation map with CelebAMask-HQ classes:
        # 1: face skin, 2: l_eyebrow, 3: r_eyebrow, 4: l_eye, 5: r_eye, 7: l_ear, 8: r_ear, 17: hair, 18: hat
        self.seg_map = np.zeros((self.h, self.w), dtype=np.int32)
        self.seg_map[:80, 40:160] = 17       # hair
        self.seg_map[90:170, 50:150] = 1     # face skin
        self.seg_map[90:98, 55:95] = 2       # left eyebrow
        self.seg_map[90:98, 105:145] = 3     # right eyebrow
        self.seg_map[102:115, 60:85] = 4     # left eye
        self.seg_map[102:115, 115:140] = 5   # right eye
        self.seg_map[110:140, 35:50] = 7     # left ear
        self.seg_map[110:140, 150:165] = 8   # right ear

        # 478 MediaPipe landmarks (synthetic 2D coordinates)
        dummy_pts = [(int(100 + 40 * np.cos(i)), int(120 + 50 * np.sin(i))) for i in range(478)]
        # Precise anchor points:
        dummy_pts[33] = (70, 108)   # left eye
        dummy_pts[133] = (80, 108)
        dummy_pts[362] = (120, 108) # right eye
        dummy_pts[263] = (130, 108)
        dummy_pts[234] = (45, 125)  # right cheek/ear
        dummy_pts[454] = (155, 125) # left cheek/ear
        dummy_pts[152] = (100, 180) # chin
        dummy_pts[10] = (100, 60)   # forehead
        dummy_pts[168] = (100, 95)  # glabella/nose bridge
        dummy_pts[70] = (55, 94)    # brow points
        dummy_pts[63] = (65, 92)
        dummy_pts[105] = (75, 90)
        dummy_pts[66] = (85, 92)
        dummy_pts[107] = (95, 94)
        dummy_pts[336] = (105, 94)
        dummy_pts[296] = (115, 92)
        dummy_pts[334] = (125, 90)
        dummy_pts[293] = (135, 92)
        dummy_pts[300] = (145, 94)

        self.landmarks = {
            "detected": True,
            "landmarks_2d": dummy_pts,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
        }

    def _create_mock_pipeline(self, color=(190, 150, 130)):
        mock_pipe = MagicMock()
        mock_gen_img = Image.new("RGB", (512, 512), color=color)
        mock_output = MagicMock()
        mock_output.images = [mock_gen_img]
        mock_pipe.return_value = mock_output
        mock_pipe.device = "cpu"
        return mock_pipe

    def test_compute_inpaint_dimensions_aspect_ratios(self):
        """Test aspect-ratio preserving dimensions divisible by 8 for diverse aspect ratios."""
        # Square 2048x2048
        w1, h1 = compute_inpaint_dimensions(2048, 2048, max_dim=768)
        self.assertEqual(w1, 768)
        self.assertEqual(h1, 768)
        self.assertEqual(w1 % 8, 0)
        self.assertEqual(h1 % 8, 0)

        # 9:16 Portrait 720x1280
        w2, h2 = compute_inpaint_dimensions(720, 1280, max_dim=768)
        self.assertEqual(h2, 768)
        self.assertEqual(w2 % 8, 0)
        self.assertAlmostEqual(w2 / h2, 720 / 1280, delta=0.05)

        # Wide Landscape 1920x1080
        w3, h3 = compute_inpaint_dimensions(1920, 1080, max_dim=768)
        self.assertEqual(w3, 768)
        self.assertEqual(h3 % 8, 0)
        self.assertAlmostEqual(w3 / h3, 1920 / 1080, delta=0.05)

    def test_target_with_hair_only(self):
        """Test clean mask construction on target with hair only (no headwear)."""
        cfg = BaldConverterConfig()
        converter = SDInpaintBaldConverter(cfg, pipeline=self._create_mock_pipeline())
        edit_mask, prot_mask, warnings = converter.build_bald_edit_mask(
            target_image=self.target_rgb,
            target_hair_mask=self.hair_mask,
            target_segmentation_map=self.seg_map,
            face_landmarks=self.landmarks,
        )
        self.assertEqual(edit_mask.size, (self.w, self.h))
        edit_np = np.asarray(edit_mask)
        prot_np = np.asarray(prot_mask)
        # Hair should be active in edit mask
        self.assertTrue(np.count_nonzero(edit_np[:60, 50:150]) > 0)
        # Eyes and ears strictly protected
        self.assertEqual(np.count_nonzero((edit_np > 0) & (prot_np > 0)), 0)

    def test_target_with_headwear_only(self):
        """Test target wearing headwear with empty hair mask."""
        cfg = BaldConverterConfig()
        converter = SDInpaintBaldConverter(cfg, pipeline=self._create_mock_pipeline())
        empty_hair = Image.new("L", (self.w, self.h), 0)
        edit_mask, prot_mask, warnings = converter.build_bald_edit_mask(
            target_image=self.target_rgb,
            target_hair_mask=empty_hair,
            target_segmentation_map=self.seg_map,
            face_landmarks=self.landmarks,
            headwear_mask=self.headwear_mask,
        )
        edit_np = np.asarray(edit_mask)
        self.assertTrue(np.count_nonzero(edit_np[:40, 40:160]) > 0)

    def test_target_with_hair_and_headwear(self):
        """Test combined removal of both hair and headwear."""
        cfg = BaldConverterConfig()
        converter = SDInpaintBaldConverter(cfg, pipeline=self._create_mock_pipeline())
        edit_mask, prot_mask, warnings = converter.build_bald_edit_mask(
            target_image=self.target_rgb,
            target_hair_mask=self.hair_mask,
            target_segmentation_map=self.seg_map,
            face_landmarks=self.landmarks,
            headwear_mask=self.headwear_mask,
        )
        edit_np = np.asarray(edit_mask)
        # Both headwear zone and hair zone are covered
        self.assertTrue(np.count_nonzero(edit_np[:40, 35:165]) > 0)
        self.assertTrue(np.count_nonzero(edit_np[40:75, 45:155]) > 0)

    def test_target_with_fringe_close_to_eyebrows(self):
        """Ensure fringe hanging close to eyebrows is editable without shaving eyebrows."""
        # Hair hangs down to y=89 (directly touching eyebrow at y=90)
        fringe_arr = np.zeros((self.h, self.w), dtype=np.uint8)
        fringe_arr[:90, 50:150] = 255
        fringe_mask = Image.fromarray(fringe_arr, mode="L")

        cfg = BaldConverterConfig()
        converter = SDInpaintBaldConverter(cfg, pipeline=self._create_mock_pipeline())
        edit_mask, prot_mask, warnings = converter.build_bald_edit_mask(
            target_image=self.target_rgb,
            target_hair_mask=fringe_mask,
            target_segmentation_map=self.seg_map,
            face_landmarks=self.landmarks,
        )
        edit_np = np.asarray(edit_mask)
        prot_np = np.asarray(prot_mask)

        # Fringe pixels at y=85-88 MUST be editable
        self.assertTrue(np.any(edit_np[85:89, 60:140] > 0))
        # Eyebrows at y=91-97 MUST be protected
        self.assertTrue(np.all(prot_np[91:97, 60:90] == 255))
        # No overlap between edit and protection
        self.assertEqual(np.count_nonzero((edit_np > 0) & (prot_np > 0)), 0)

    def test_target_with_hair_close_to_ears(self):
        """Ensure sideburns/hair near ears do not erode the ear cartilage."""
        cfg = BaldConverterConfig()
        converter = SDInpaintBaldConverter(cfg, pipeline=self._create_mock_pipeline())
        edit_mask, prot_mask, warnings = converter.build_bald_edit_mask(
            target_image=self.target_rgb,
            target_hair_mask=self.hair_mask,
            target_segmentation_map=self.seg_map,
            face_landmarks=self.landmarks,
        )
        prot_np = np.asarray(prot_mask)
        edit_np = np.asarray(edit_mask)
        # Ears (classes 7 and 8) MUST be 100% in protection mask
        self.assertTrue(np.all(prot_np[110:140, 35:50] == 255))
        self.assertTrue(np.all(prot_np[110:140, 150:165] == 255))
        self.assertEqual(np.count_nonzero(edit_np[110:140, 35:50]), 0)

    def test_rgba_target_input(self):
        """Test that RGBA targets preserve their original alpha channel."""
        cfg = BaldConverterConfig(num_inference_steps=2)
        converter = SDInpaintBaldConverter(cfg, pipeline=self._create_mock_pipeline())
        result = converter.convert(
            target_image=self.target_rgba,
            target_hair_mask=self.hair_mask,
            target_segmentation_map=self.seg_map,
            face_landmarks=self.landmarks,
        )
        self.assertEqual(result.bald_base.mode, "RGBA")
        self.assertEqual(result.bald_base.size, (self.w, self.h))
        # Check alpha channel preserved
        orig_a = np.asarray(self.target_rgba)[:, :, 3]
        out_a = np.asarray(result.bald_base)[:, :, 3]
        self.assertTrue(np.array_equal(orig_a, out_a))

    def test_missing_or_failed_landmarks(self):
        """Test graceful fallback when landmarks are completely missing or detected=False."""
        cfg = BaldConverterConfig()
        converter = SDInpaintBaldConverter(cfg, pipeline=self._create_mock_pipeline())

        # None landmarks
        edit_mask, prot_mask, warnings = converter.build_bald_edit_mask(
            target_image=self.target_rgb,
            target_hair_mask=self.hair_mask,
            target_segmentation_map=self.seg_map,
            face_landmarks=None,
        )
        self.assertEqual(edit_mask.size, (self.w, self.h))

        # detected=False landmarks
        edit_mask2, prot_mask2, warnings2 = converter.build_bald_edit_mask(
            target_image=self.target_rgb,
            target_hair_mask=self.hair_mask,
            target_segmentation_map=self.seg_map,
            face_landmarks={"detected": False, "landmarks_2d": []},
        )
        self.assertEqual(edit_mask2.size, (self.w, self.h))

    def test_manual_headwear_mask_override(self):
        """Test that explicit manual headwear mask is honored."""
        cfg = BaldConverterConfig()
        converter = SDInpaintBaldConverter(cfg, pipeline=self._create_mock_pipeline())
        custom_hw = Image.new("L", (self.w, self.h), 0)
        custom_hw_arr = np.asarray(custom_hw).copy()
        custom_hw_arr[10:30, 20:60] = 255
        custom_hw = Image.fromarray(custom_hw_arr, mode="L")

        edit_mask, prot_mask, warnings = converter.build_bald_edit_mask(
            target_image=self.target_rgb,
            target_hair_mask=self.hair_mask,
            target_segmentation_map=self.seg_map,
            face_landmarks=self.landmarks,
            headwear_mask=custom_hw,
        )
        edit_np = np.asarray(edit_mask)
        # Custom mask region must be active
        self.assertTrue(np.any(edit_np[15:25, 25:55] > 0))

    def test_empty_and_large_mask_diagnostics(self):
        """Test diagnostic warnings for empty mask and implausibly large mask."""
        cfg = BaldConverterConfig()
        converter = SDInpaintBaldConverter(cfg, pipeline=self._create_mock_pipeline())

        # Empty mask
        empty_mask = Image.new("L", (self.w, self.h), 0)
        _, _, w_empty = converter.build_bald_edit_mask(self.target_rgb, empty_mask)
        self.assertTrue(any("0 active pixels" in w for w in w_empty))

        # Implausibly large mask (>85%)
        huge_mask = Image.new("L", (self.w, self.h), 255)
        _, _, w_huge = converter.build_bald_edit_mask(self.target_rgb, huge_mask)
        self.assertTrue(any("covers" in w for w in w_huge))

    def test_exact_preservation_outside_edit_mask(self):
        """Verify 100.00% bit-exact pixel preservation outside edit mask in convert()."""
        cfg = BaldConverterConfig(num_inference_steps=2, seed=42)
        mock_pipe = self._create_mock_pipeline(color=(120, 80, 60))
        converter = SDInpaintBaldConverter(cfg, pipeline=mock_pipe)

        result = converter.convert(
            target_image=self.target_rgb,
            target_hair_mask=self.hair_mask,
            target_segmentation_map=self.seg_map,
            face_landmarks=self.landmarks,
        )

        target_np = np.asarray(self.target_rgb)
        bald_np = np.asarray(result.bald_base)[:, :, :3]
        edit_np = np.asarray(result.edit_mask)

        outside = edit_np == 0
        diff = np.abs(bald_np[outside].astype(int) - target_np[outside].astype(int))
        self.assertEqual(int(np.max(diff)), 0, "Outside pixels were corrupted!")

    def test_stage1_quality_gate_evaluator(self):
        """Test Quality Gate evaluator logic across passing and failing conditions."""
        target_np = np.asarray(self.target_rgb)
        bald_np = target_np.copy()
        edit_np = np.asarray(self.hair_mask)
        prot_np = np.zeros((self.h, self.w), dtype=np.uint8)

        # Inpaint with plausible random texture inside edit mask
        bald_np[edit_np > 128] = np.random.randint(150, 210, size=bald_np[edit_np > 128].shape, dtype=np.uint8)

        qg = evaluate_stage1_quality_gate(
            original_target=target_np,
            bald_base=bald_np,
            edit_mask=edit_np,
            protection_mask=prot_np,
        )
        self.assertEqual(qg["status"], "PASSED")
        self.assertEqual(qg["preservation_pct"], 100.0)
        self.assertEqual(qg["max_outside_diff"], 0)


if __name__ == "__main__":
    unittest.main()
