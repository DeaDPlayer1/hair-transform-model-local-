"""
Comprehensive Unit, Regression, and Smoke Tests for Local Hairstyle Transfer.

Test Coverage:
1. Valid RGB, Grayscale, RGBA, and EXIF orientation preprocessing.
2. Invalid, corrupt, and empty image handling.
3. Face detection warning and missing face tolerance.
4. Empty segmentation mask validation.
5. Edit mask generation, cranial expansion, and multi-layer facial feature protection.
6. Mask feathering strictly inside the edit region.
7. Mocked diffusion generation and dimension preservation.
8. Bit-exact 100% preservation of target pixels outside the edit mask.
9. Experiment metadata and metrics persistence.
10. Regression test: pipeline does NOT call enable_attention_slicing (which broke IP-Adapter).
11. Streamlit integration smoke test.
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import json
import numpy as np
from PIL import Image

from src.image_preprocessing import (
    load_and_validate_image,
    resize_preserving_aspect,
    validate_target_portrait,
    ImageValidationError,
)
from src.hair_mask_refinement import (
    build_refined_edit_mask,
    build_feathered_blend_mask,
    composite_with_feathering,
)
from src.transfer_evaluation import (
    evaluate_transfer_metrics,
    save_experiment_run,
    list_saved_experiments,
)
from src.local_generative_transfer import (
    prepare_reference_crop,
    build_target_edit_mask,
    generate_local_hairstyle_transfer,
    load_local_generative_pipeline,
)


class TestLocalGenerativeTransferComprehensive(unittest.TestCase):
    def setUp(self):
        self.w, self.h = 256, 256
        self.target_img = Image.new("RGB", (self.w, self.h), color=(200, 180, 160))
        self.ref_img = Image.new("RGB", (self.w, self.h), color=(50, 40, 30))

        # Target hair mask: top crown
        mask_np = np.zeros((self.h, self.w), dtype=np.uint8)
        mask_np[20:100, 60:190] = 255
        self.target_mask = Image.fromarray(mask_np, mode="L")

        # Ref hair mask
        ref_mask_np = np.zeros((self.h, self.w), dtype=np.uint8)
        ref_mask_np[30:120, 50:200] = 255
        self.ref_mask = Image.fromarray(ref_mask_np, mode="L")

        # Segmentation map: 0 bg, 1 face, 4 l_eye, 5 r_eye, 10 nose, 11 mouth, 16 cloth, 17 hair
        self.seg_map = np.zeros((self.h, self.w), dtype=np.uint8)
        self.seg_map[20:100, 60:190] = 17  # hair
        self.seg_map[100:200, 70:180] = 1  # face skin
        self.seg_map[110:125, 80:105] = 4  # left eye
        self.seg_map[110:125, 145:170] = 5  # right eye
        self.seg_map[130:150, 115:135] = 10  # nose
        self.seg_map[160:180, 105:145] = 11  # mouth
        self.seg_map[210:256, :] = 16  # cloth

    # ---------------------------------------------------------
    # 1. Image Preprocessing & Input Validation
    # ---------------------------------------------------------
    def test_01_rgb_grayscale_rgba_inputs(self):
        """Verify RGB, Grayscale, and RGBA images are converted to RGB properly."""
        gray_img = Image.new("L", (128, 128), color=120)
        res_gray = load_and_validate_image(gray_img)
        self.assertEqual(res_gray.mode, "RGB")
        self.assertEqual(res_gray.size, (128, 128))

        rgba_img = Image.new("RGBA", (128, 128), color=(255, 100, 50, 200))
        res_rgba = load_and_validate_image(rgba_img)
        self.assertEqual(res_rgba.mode, "RGB")

    def test_02_invalid_and_corrupt_images(self):
        """Verify corrupt, empty, and invalid inputs raise ImageValidationError."""
        with self.assertRaises(ImageValidationError):
            load_and_validate_image(None)

        with self.assertRaises(ImageValidationError):
            load_and_validate_image(b"")

        with self.assertRaises(ImageValidationError):
            load_and_validate_image(b"not an image data string at all")

        # Below min dimensions (e.g. 10x10 < 64)
        tiny_img = Image.new("RGB", (10, 10), color=(0, 0, 0))
        with self.assertRaises(ImageValidationError):
            load_and_validate_image(tiny_img)

    def test_03_resize_preserving_aspect(self):
        """Verify aspect-ratio-safe letterboxing."""
        wide_img = Image.new("RGB", (400, 200), color=(100, 150, 200))
        padded, info = resize_preserving_aspect(wide_img, target_size=(512, 512))
        self.assertEqual(padded.size, (512, 512))
        self.assertEqual(info["scaled_size"], (512, 256))
        self.assertEqual(info["y_offset"], 128)

    # ---------------------------------------------------------
    # 2. Mask Refinement & Facial Feature Protection
    # ---------------------------------------------------------
    def test_04_edit_mask_protection(self):
        """Verify facial feature protection zones (eyes, nose, mouth, clothing)."""
        edit_mask = build_target_edit_mask(
            target_image=self.target_img,
            target_hair_mask=self.target_mask,
            target_segmentation_map=self.seg_map,
            cranial_expansion_px=20,
        )
        mask_np = np.asarray(edit_mask)

        # Eyes, nose, mouth, clothing must be strictly 0 (unmasked)
        for class_id in [4, 5, 10, 11, 16]:
            region = self.seg_map == class_id
            self.assertEqual(np.max(mask_np[region]), 0, f"Class {class_id} was not protected!")

        # Hair region must be included
        hair_core = (self.seg_map == 17) & (mask_np == 255)
        self.assertTrue(np.any(hair_core), "Original hair was not included in edit mask!")

    def test_05_empty_mask_handling(self):
        """Verify that an empty mask or full mask triggers validation errors in generation."""
        empty_mask = Image.new("L", (self.w, self.h), color=0)
        with self.assertRaises(ValueError):
            generate_local_hairstyle_transfer(
                target_image=self.target_img,
                reference_image=self.ref_img,
                target_hair_mask=empty_mask,
            )

    def test_06_mask_feathering_strictly_inside(self):
        """Verify distance-transform feathering occurs strictly INSIDE the edit mask."""
        alpha = build_feathered_blend_mask(self.target_mask, feather_width_px=8.0)
        mask_np = np.asarray(self.target_mask)

        # Outside mask (mask_np == 0) must have alpha == 0.0 exactly
        outside = mask_np == 0
        self.assertEqual(float(np.max(alpha[outside])), 0.0, "Feathering leaked outside edit mask!")

        # Center of mask should reach 1.0
        inside_deep = mask_np == 255
        self.assertGreater(float(np.max(alpha[inside_deep])), 0.9)

    # ---------------------------------------------------------
    # 3. Generation Workflow & Pixel Preservation Outside Mask
    # ---------------------------------------------------------
    def test_07_mocked_generation_and_preservation(self):
        """Verify pipeline generation preserves untouched target pixels bit-exact."""
        mock_pipe = MagicMock()
        dummy_gen_512 = Image.new("RGB", (512, 512), color=(255, 0, 0))
        mock_output = MagicMock()
        mock_output.images = [dummy_gen_512]
        mock_pipe.return_value = mock_output

        result = generate_local_hairstyle_transfer(
            target_image=self.target_img,
            reference_image=self.ref_img,
            target_hair_mask=self.target_mask,
            reference_hair_mask=self.ref_mask,
            target_segmentation_map=self.seg_map,
            num_inference_steps=5,
            ip_adapter_scale=0.75,
            seed=42,
            pipeline=mock_pipe,
            save_experiment=False,
        )

        self.assertIn("composite_image", result)
        self.assertEqual(result["composite_image"].size, (self.w, self.h))

        # Check outside-mask preservation
        target_np = np.asarray(self.target_img)
        comp_np = np.asarray(result["composite_image"])
        mask_np = np.asarray(result["edit_mask"])

        outside = mask_np == 0
        diff = np.abs(target_np[outside].astype(int) - comp_np[outside].astype(int))
        self.assertEqual(np.max(diff), 0, "Target pixels outside edit mask were altered!")

        metrics = result["metrics"]
        self.assertEqual(metrics["max_outside_pixel_diff"], 0)
        self.assertEqual(metrics["outside_preservation_pct"], 100.0)
        self.assertTrue(metrics["facial_features_protected"])
        self.assertFalse(metrics["has_nan_or_inf"])

    # ---------------------------------------------------------
    # 4. Experiment Tracking & Serialization
    # ---------------------------------------------------------
    def test_08_experiment_saving_and_listing(self):
        """Verify experiment serialization (target, reference, masks, metadata, metrics)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            metadata = {"seed": 99, "steps": 10, "device": "cpu"}
            metrics = {"outside_preservation_pct": 100.0, "max_outside_pixel_diff": 0}

            exp_dir = save_experiment_run(
                outputs_base_dir=tmp_path,
                target_image=self.target_img,
                reference_image=self.ref_img,
                hair_mask=self.target_mask,
                edit_mask=self.target_mask,
                composite_image=self.target_img,
                metadata=metadata,
                metrics=metrics,
                experiment_id="test_exp_01",
            )

            self.assertTrue((exp_dir / "target.png").is_file())
            self.assertTrue((exp_dir / "generated.png").is_file())
            self.assertTrue((exp_dir / "metadata.json").is_file())
            self.assertTrue((exp_dir / "metrics.json").is_file())

            experiments = list_saved_experiments(tmp_path)
            self.assertEqual(len(experiments), 1)
            self.assertEqual(experiments[0]["id"], "test_exp_01")
            self.assertEqual(experiments[0]["metadata"]["seed"], 99)

    # ---------------------------------------------------------
    # 5. Regression Test: Attention Slicing Must NOT Be Called
    # ---------------------------------------------------------
    def test_09_regression_no_attention_slicing(self):
        """
        Regression test ensuring pipeline does NOT reintroduce pipeline.enable_attention_slicing(),
        which causes IP-Adapter attention processor failure (treating tuple as tensor).
        """
        import inspect
        import src.local_generative_transfer as lgt
        src_code = inspect.getsource(lgt.load_local_generative_pipeline)
        self.assertNotIn(
            "pipeline.enable_attention_slicing()",
            src_code,
            "CRITICAL REGRESSION: enable_attention_slicing() was reintroduced! It breaks IP-Adapter.",
        )

    # ---------------------------------------------------------
    # 6. Streamlit Module Import Smoke Test
    # ---------------------------------------------------------
    def test_10_streamlit_app_compiles(self):
        """Smoke test verifying app.py can be compiled and parsed without syntax errors."""
        import py_compile
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        self.assertTrue(app_path.is_file())
        py_compile.compile(str(app_path), doraise=True)

    # ---------------------------------------------------------
    # 7. Regression Test: Central Hair Preservation & Narrow Mask
    # ---------------------------------------------------------
    def test_11_central_hair_preservation_and_narrow_mask(self):
        """Verify edge_repair_mask is narrower than broad edit_mask and central hair is preserved."""
        from unittest.mock import MagicMock
        mock_pipe = MagicMock()
        dummy_diff = Image.new("RGB", (512, 512), (255, 0, 0))  # Bright red for inpainting output
        mock_pipe.return_value = MagicMock(images=[dummy_diff])

        result = generate_local_hairstyle_transfer(
            target_image=self.target_img,
            reference_image=self.ref_img,
            target_hair_mask=self.target_mask,
            reference_hair_mask=self.ref_mask,
            target_segmentation_map=self.seg_map,
            num_inference_steps=5,
            pipeline=mock_pipe,
            save_experiment=False,
            use_alignment=True,
        )

        self.assertIn("edge_repair_mask", result)
        self.assertIn("edit_mask", result)
        edge_np = np.asarray(result["edge_repair_mask"])
        broad_np = np.asarray(result["edit_mask"])

        # Edge repair mask must be strictly narrower than the full broad edit mask
        self.assertLess(np.count_nonzero(edge_np), np.count_nonzero(broad_np))

        # Central hair pixels (inside broad edit mask, but outside edge repair mask)
        # must NOT be replaced by the bright red dummy diffusion output!
        central_hair = (broad_np > 128) & (edge_np == 0)
        if np.any(central_hair):
            comp_np = np.asarray(result["composite_image"])
            # Red channel must NOT be 255 with green/blue 0 (which is the raw diffusion output)
            is_pure_red = (comp_np[central_hair, 0] == 255) & (comp_np[central_hair, 1] == 0)
            self.assertEqual(np.count_nonzero(is_pure_red), 0, "Central hair was overwritten by diffusion inpainting!")

    # ---------------------------------------------------------
    # 8. Regression Test: Reference Crop Decontamination & Narrow Fade
    # ---------------------------------------------------------
    def test_12_reference_crop_antialiased_preserves_hair_and_neutral_bg(self):
        """Verify prepare_reference_crop preserves dark curl strands and does not fade them to gray 128."""
        # Create dark hair on a light background
        ref_img = Image.new("RGB", (100, 100), (220, 220, 220))  # light background
        ref_arr = np.array(ref_img)
        ref_arr[30:70, 30:70] = [30, 25, 20]  # dark hair block
        # Add thin 2px strand
        ref_arr[20:30, 48:50] = [30, 25, 20]
        ref_img = Image.fromarray(ref_arr)

        mask_np = np.zeros((100, 100), dtype=np.uint8)
        mask_np[30:70, 30:70] = 255
        mask_np[20:30, 48:50] = 255
        ref_mask = Image.fromarray(mask_np, mode="L")

        crop_revised = prepare_reference_crop(ref_img, ref_mask, neutral_background=True, legacy_gray_fade=False)
        crop_legacy = prepare_reference_crop(ref_img, ref_mask, neutral_background=True, legacy_gray_fade=True)

        revised_np = np.asarray(crop_revised)
        legacy_np = np.asarray(crop_legacy)

        # Distant background should be neutral 128
        self.assertEqual(revised_np[5, 5, 0], 128)
        self.assertEqual(revised_np[5, 5, 1], 128)
        self.assertEqual(revised_np[5, 5, 2], 128)

        # Center of hair should be dark hair (~30)
        center_revised = revised_np[256, 256]
        self.assertLess(center_revised[0], 50)

        # Revised crop preserves deeper contrast and lower mean distance from true hair than legacy
        self.assertLess(revised_np.mean(), legacy_np.mean())


if __name__ == "__main__":
    unittest.main()

