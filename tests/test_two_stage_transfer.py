"""
Unit & Regression Tests for Two-Stage Hairstyle Transfer Orchestration.
Fast execution using mocked model pipelines.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from src.two_stage_transfer import (
    generate_two_stage_hairstyle_transfer,
    load_two_stage_config,
    compute_preservation_metrics,
)
from src.bald_converter import BaldConversionResult


class TestTwoStageTransfer(unittest.TestCase):
    def setUp(self):
        # Target image: 150x150 distinct color
        self.target_img = Image.new("RGB", (150, 150), color=(210, 180, 140))
        # Hair mask on top 40%
        hair_arr = np.zeros((150, 150), dtype=np.uint8)
        hair_arr[:60, 20:130] = 255
        self.target_hair_mask = Image.fromarray(hair_arr, mode="L")

        # Reference image: 100x120 (different resolution)
        self.ref_img = Image.new("RGB", (100, 120), color=(50, 40, 30))
        ref_hair_arr = np.zeros((120, 100), dtype=np.uint8)
        ref_hair_arr[:50, 10:90] = 255
        self.ref_hair_mask = Image.fromarray(ref_hair_arr, mode="L")

        self.seg_map = np.zeros((150, 150), dtype=np.int32)
        self.seg_map[70:130, 40:110] = 2  # skin
        self.seg_map[80:100, 20:40] = 7  # l_ear
        self.seg_map[80:100, 110:130] = 8  # r_ear

        self.target_lms = {
            "detected": True,
            "landmarks_2d": [(40, 70), (110, 70), (115, 120), (75, 140), (35, 120)],
        }
        self.ref_lms = {
            "detected": True,
            "landmarks_2d": [(25, 60), (75, 60), (80, 100), (50, 110), (20, 100)],
        }

    def _create_mock_pipeline(self):
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.images = [Image.new("RGB", (512, 512), color=(150, 120, 90))]
        mock_pipe.return_value = mock_output
        mock_pipe.device = "cpu"
        mock_pipe.image_encoder = None
        return mock_pipe

    def test_stage2_receives_bald_base_and_preserves_dimensions(self):
        mock_pipe = self._create_mock_pipeline()

        # Track what image align_reference_to_target received
        with patch("src.two_stage_transfer.align_reference_to_target") as mock_align:
            mock_align.return_value = {
                "aligned_hair_rgb": np.zeros((150, 150, 3), dtype=np.uint8),
                "aligned_hair_alpha": np.zeros((150, 150), dtype=np.float32),
                "pasted_composite": Image.new("RGB", (150, 150), color=(190, 160, 130)),
                "edge_repair_mask": Image.new("L", (150, 150), color=128),
                "broad_edit_mask": Image.new("L", (150, 150), color=255),
                "hard_protection": np.zeros((150, 150), dtype=np.uint8),
                "identity_protection": np.zeros((150, 150), dtype=np.uint8),
                "warnings": [],
            }

            result = generate_two_stage_hairstyle_transfer(
                target_image=self.target_img,
                reference_image=self.ref_img,
                pipeline=mock_pipe,
                target_hair_mask=self.target_hair_mask,
                reference_hair_mask=self.ref_hair_mask,
                target_segmentation_map=self.seg_map,
                face_landmarks=self.target_lms,
                reference_landmarks=self.ref_lms,
                save_experiment=False,
            )

            # Assertion 1: Stage 2 received the bald base, NOT the original target
            called_target = mock_align.call_args[1]["target_image"]
            self.assertEqual(called_target, result["bald_base"])

            # Assertion 7: Output retains original target dimensions
            self.assertEqual(result["composite_image"].size, (150, 150))
            self.assertEqual(result["bald_base"].size, (150, 150))

    def test_original_input_images_are_never_mutated(self):
        mock_pipe = self._create_mock_pipeline()
        orig_tgt_bytes = self.target_img.tobytes()
        orig_ref_bytes = self.ref_img.tobytes()

        with patch("src.two_stage_transfer.align_reference_to_target") as mock_align:
            mock_align.return_value = {
                "aligned_hair_rgb": np.zeros((150, 150, 3), dtype=np.uint8),
                "aligned_hair_alpha": np.zeros((150, 150), dtype=np.float32),
                "pasted_composite": Image.new("RGB", (150, 150), color=(190, 160, 130)),
                "edge_repair_mask": Image.new("L", (150, 150), color=0),
                "broad_edit_mask": Image.new("L", (150, 150), color=255),
                "hard_protection": np.zeros((150, 150), dtype=np.uint8),
                "identity_protection": np.zeros((150, 150), dtype=np.uint8),
                "warnings": [],
            }

            generate_two_stage_hairstyle_transfer(
                target_image=self.target_img,
                reference_image=self.ref_img,
                pipeline=mock_pipe,
                target_hair_mask=self.target_hair_mask,
                reference_hair_mask=self.ref_hair_mask,
                target_segmentation_map=self.seg_map,
                face_landmarks=self.target_lms,
                reference_landmarks=self.ref_lms,
                save_experiment=False,
            )

            self.assertEqual(self.target_img.tobytes(), orig_tgt_bytes, "Target image was mutated!")
            self.assertEqual(self.ref_img.tobytes(), orig_ref_bytes, "Reference image was mutated!")

    def test_outside_broad_envelope_exact_restoration(self):
        # Create a broad envelope covering only the top half
        broad_np = np.zeros((150, 150), dtype=np.uint8)
        broad_np[:75, :] = 255
        broad_pil = Image.fromarray(broad_np, mode="L")

        mock_pipe = self._create_mock_pipeline()

        with patch("src.two_stage_transfer.align_reference_to_target") as mock_align:
            # Let diffusion return radically different color everywhere
            mock_align.return_value = {
                "aligned_hair_rgb": np.full((150, 150, 3), 50, dtype=np.uint8),
                "aligned_hair_alpha": np.ones((150, 150), dtype=np.float32),
                "pasted_composite": Image.new("RGB", (150, 150), color=(50, 50, 50)),
                "edge_repair_mask": Image.new("L", (150, 150), color=255),
                "broad_edit_mask": broad_pil,
                "hard_protection": np.zeros((150, 150), dtype=np.uint8),
                "identity_protection": np.zeros((150, 150), dtype=np.uint8),
                "warnings": [],
            }

            res = generate_two_stage_hairstyle_transfer(
                target_image=self.target_img,
                reference_image=self.ref_img,
                pipeline=mock_pipe,
                target_hair_mask=self.target_hair_mask,
                reference_hair_mask=self.ref_hair_mask,
                target_segmentation_map=self.seg_map,
                face_landmarks=self.target_lms,
                reference_landmarks=self.ref_lms,
                save_experiment=False,
            )

            final_np = np.asarray(res["composite_image"])
            target_np = np.asarray(self.target_img)

            # Bottom half (outside broad envelope) must be 100% bit-exact to original target
            bottom_diff = np.abs(final_np[75:, :].astype(int) - target_np[75:, :].astype(int))
            self.assertEqual(np.max(bottom_diff), 0, "Outside envelope pixels were corrupted!")
            self.assertEqual(res["metrics"]["outside_pixel_preservation_pct"], 100.0)

    def test_identity_protection_pixels_are_strictly_preserved(self):
        # Mark ears as identity protection
        ident_np = np.zeros((150, 150), dtype=np.uint8)
        ident_np[80:100, 20:40] = 255  # l_ear
        ident_np[80:100, 110:130] = 255  # r_ear

        mock_pipe = self._create_mock_pipeline()

        with patch("src.two_stage_transfer.align_reference_to_target") as mock_align:
            mock_align.return_value = {
                "aligned_hair_rgb": np.full((150, 150, 3), 255, dtype=np.uint8),
                "aligned_hair_alpha": np.ones((150, 150), dtype=np.float32),
                "pasted_composite": Image.new("RGB", (150, 150), color=(255, 0, 0)),
                "edge_repair_mask": Image.new("L", (150, 150), color=255),
                "broad_edit_mask": Image.new("L", (150, 150), color=255),
                "hard_protection": ident_np,
                "identity_protection": ident_np,
                "warnings": [],
            }

            res = generate_two_stage_hairstyle_transfer(
                target_image=self.target_img,
                reference_image=self.ref_img,
                pipeline=mock_pipe,
                target_hair_mask=self.target_hair_mask,
                reference_hair_mask=self.ref_hair_mask,
                target_segmentation_map=self.seg_map,
                face_landmarks=self.target_lms,
                reference_landmarks=self.ref_lms,
                save_experiment=False,
            )

            final_np = np.asarray(res["composite_image"])
            target_np = np.asarray(self.target_img)

            # Identity protection pixels must be bit-exact
            self.assertEqual(res["metrics"]["identity_protection_leak_count"], 0)
            ear_diff = np.abs(final_np[ident_np > 0].astype(int) - target_np[ident_np > 0].astype(int))
            self.assertEqual(np.max(ear_diff), 0)

    def test_failed_stage1_does_not_proceed_silently(self):
        mock_pipe = MagicMock()
        # Mock SD inpaint raising an error or returning invalid
        mock_pipe.side_effect = RuntimeError("GPU out of memory in Stage 1")

        with self.assertRaises(RuntimeError):
            generate_two_stage_hairstyle_transfer(
                target_image=self.target_img,
                reference_image=self.ref_img,
                pipeline=mock_pipe,
                target_hair_mask=self.target_hair_mask,
                save_experiment=False,
            )

    def test_preservation_metrics_helper(self):
        tgt = np.full((10, 10, 3), 100, dtype=np.uint8)
        comp = tgt.copy()
        envelope = np.zeros((10, 10), dtype=np.uint8)
        # Inside envelope, let it differ
        envelope[:5, :] = 255
        comp[:5, :] = 200

        metrics = compute_preservation_metrics(tgt, comp, envelope)
        self.assertEqual(metrics["outside_pixel_preservation_pct"], 100.0)
        self.assertEqual(metrics["max_outside_pixel_diff"], 0.0)

    def test_combined_envelope_prevents_old_target_hair_restoration(self):
        """
        Regression test: When Stage 1 clears target hair over a region that Stage 2
        broad edit mask does not cover (e.g. shaving long hair into short hair),
        the final outside-envelope restoration MUST NOT restore the old target hair.
        """
        w, h = 100, 100
        # Target has long hair down to y=80
        target = Image.new("RGB", (w, h), color=(150, 150, 150))
        target_np = np.asarray(target).copy()
        # Old target hair is black (0, 0, 0) in y in [40, 80]
        target_np[40:80, 20:80] = [0, 0, 0]
        target_img = Image.fromarray(target_np)

        target_hair = np.zeros((h, w), dtype=np.uint8)
        target_hair[40:80, 20:80] = 255
        target_hair_mask = Image.fromarray(target_hair, mode="L")

        # Stage 1 converts target hair to bald scalp skin color (210, 180, 140)
        bald_np = np.full((h, w, 3), 150, dtype=np.uint8)
        bald_np[40:80, 20:80] = [210, 180, 140]  # Bald scalp skin
        bald_base = Image.fromarray(bald_np)

        # Stage 1 edit mask covered all target hair
        s1_edit_mask = Image.fromarray(target_hair, mode="L")

        # Stage 2 reference hair is very short, only covering y in [20, 50]
        # So y in [51, 80] is outside Stage 2 broad mask!
        s2_broad = np.zeros((h, w), dtype=np.uint8)
        s2_broad[20:50, 25:75] = 255
        s2_broad_mask = Image.fromarray(s2_broad, mode="L")

        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.images = [Image.new("RGB", (512, 512), color=(50, 40, 30))]
        mock_pipe.return_value = mock_output

        with patch("src.two_stage_transfer.create_bald_converter") as mock_create_converter, \
             patch("src.two_stage_transfer.align_reference_to_target") as mock_align:

            mock_conv = MagicMock()
            mock_conv.convert.return_value = MagicMock(
                bald_base=bald_base,
                edit_mask=s1_edit_mask,
                protection_mask=Image.new("L", (w, h), 0),
                raw_generated=bald_base,
                warnings=[],
                metadata={},
            )
            mock_create_converter.return_value = mock_conv

            mock_align.return_value = {
                "aligned_hair_rgb": np.zeros((h, w, 3), dtype=np.uint8),
                "aligned_hair_alpha": np.zeros((h, w), dtype=np.float32),
                "pasted_composite": bald_base,
                "edge_repair_mask": Image.new("L", (w, h), 0),
                "broad_edit_mask": s2_broad_mask,
                "hard_protection": None,
                "identity_protection": None,
                "warnings": [],
            }

            res = generate_two_stage_hairstyle_transfer(
                target_image=target_img,
                reference_image=self.ref_img,
                pipeline=mock_pipe,
                target_hair_mask=target_hair_mask,
                save_experiment=False,
            )

            final_np = np.asarray(res["composite_image"])
            # At y=65, x=50:
            # - Old target hair was black [0, 0, 0]
            # - Stage 1 bald scalp is skin [210, 180, 140]
            # - Stage 2 broad mask did NOT cover y=65
            # With combined envelope restoration, the bald skin [210, 180, 140] MUST be preserved!
            # The black old target hair [0, 0, 0] MUST NOT be restored!
            pixel_at_shaved_area = final_np[65, 50]
            self.assertEqual(
                list(pixel_at_shaved_area),
                [210, 180, 140],
                f"Old target hair was restored over shaved area! Pixel is {pixel_at_shaved_area}",
            )


if __name__ == "__main__":
    unittest.main()
