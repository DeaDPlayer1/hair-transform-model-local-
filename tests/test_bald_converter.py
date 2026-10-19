"""
Unit tests for Stage 1 Bald Converter (fast, mocked backends, no real diffusion).
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
)


class TestBaldConverter(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 200x200 target image
        self.target_img = Image.new("RGB", (200, 200), color=(220, 200, 180))
        # Hair on upper half
        hair_arr = np.zeros((200, 200), dtype=np.uint8)
        hair_arr[:80, 40:160] = 255
        self.hair_mask = Image.fromarray(hair_arr, mode="L")

        # Segmentation map with skin=2, ears=7, 8, eyes=5, 6
        self.seg_map = np.zeros((200, 200), dtype=np.int32)
        self.seg_map[90:170, 50:150] = 2  # skin
        self.seg_map[110:140, 30:50] = 7  # left ear
        self.seg_map[110:140, 150:170] = 8  # right ear
        self.seg_map[100:115, 70:90] = 5  # left eye
        self.seg_map[100:115, 110:130] = 6  # right eye

        self.landmarks = {
            "detected": True,
            "landmarks_2d": [(50, 90), (150, 90), (160, 150), (100, 190), (40, 150)],
        }

    def test_config_from_dict_and_defaults(self):
        cfg = BaldConverterConfig.from_dict({
            "backend": "sd-inpaint",
            "num_inference_steps": 10,
            "guidance_scale": 8.0,
            "unknown_extra_key": "should_be_ignored",
        })
        self.assertEqual(cfg.backend, "sd-inpaint")
        self.assertEqual(cfg.num_inference_steps, 10)
        self.assertEqual(cfg.guidance_scale, 8.0)
        self.assertEqual(cfg.seed, 42)

    def test_build_bald_edit_mask_dimensions_and_protection(self):
        cfg = BaldConverterConfig(max_hair_mask_expansion_px=10)
        converter = SDInpaintBaldConverter(cfg, pipeline=MagicMock())

        # Test with mismatched mask dimension
        small_mask = self.hair_mask.resize((100, 100))
        edit_mask, prot_mask, warnings = converter.build_bald_edit_mask(
            target_image=self.target_img,
            target_hair_mask=small_mask,
            target_segmentation_map=self.seg_map,
            face_landmarks=self.landmarks,
        )

        self.assertEqual(edit_mask.size, (200, 200))
        self.assertEqual(prot_mask.size, (200, 200))

        edit_np = np.asarray(edit_mask)
        prot_np = np.asarray(prot_mask)

        # Ears and eyes must be in protection mask
        self.assertTrue(np.all(prot_np[110:140, 30:50] == 255))  # Left ear
        self.assertTrue(np.all(prot_np[110:140, 150:170] == 255))  # Right ear
        # Protected pixels must never be in edit mask
        self.assertEqual(np.count_nonzero((edit_np > 0) & (prot_np > 0)), 0)

    def test_outside_pixels_strictly_preserved_in_sd_converter(self):
        cfg = BaldConverterConfig(num_inference_steps=2, seed=123)

        # Create mock pipeline
        mock_pipe = MagicMock()
        # Mock diffusion output: synthetic generated bald image with distinct color
        mock_gen_img = Image.new("RGB", (512, 512), color=(180, 140, 120))
        mock_output = MagicMock()
        mock_output.images = [mock_gen_img]
        mock_pipe.return_value = mock_output
        mock_pipe.device = "cpu"

        converter = SDInpaintBaldConverter(cfg, pipeline=mock_pipe)
        result = converter.convert(
            target_image=self.target_img,
            target_hair_mask=self.hair_mask,
            target_segmentation_map=self.seg_map,
            face_landmarks=self.landmarks,
        )

        self.assertIsInstance(result, BaldConversionResult)
        self.assertEqual(result.bald_base.size, (200, 200))

        target_np = np.asarray(self.target_img)
        bald_np = np.asarray(result.bald_base)
        edit_np = np.asarray(result.edit_mask)

        # Pixels outside edit mask MUST be 100% bit-exact identical
        outside_mask = edit_np == 0
        diff = np.abs(bald_np[outside_mask].astype(int) - target_np[outside_mask].astype(int))
        self.assertEqual(np.max(diff), 0, "Pixels outside edit mask were modified!")

    def test_invalid_diffusion_output_raises_value_error(self):
        cfg = BaldConverterConfig()
        mock_pipe = MagicMock()
        # Return a solid black (blank) image std == 0
        blank_img = Image.new("RGB", (512, 512), color=(0, 0, 0))
        mock_pipe.return_value.images = [blank_img]
        mock_pipe.device = "cpu"

        converter = SDInpaintBaldConverter(cfg, pipeline=mock_pipe)
        with self.assertRaises(ValueError):
            converter.convert(self.target_img, self.hair_mask)

    def test_missing_onnx_model_handling(self):
        # Configure non-existent onnx path
        cfg = BaldConverterConfig(backend="onnx", onnx_model_path="models/non_existent.onnx")
        with self.assertRaises(FileNotFoundError):
            create_bald_converter(cfg)

    def test_create_bald_converter_auto_fallback(self):
        # Auto backend with no onnx file should cleanly fall back to SD inpainting
        cfg = BaldConverterConfig(backend="auto", onnx_model_path=None)
        mock_pipe = MagicMock()
        converter = create_bald_converter(cfg, pipeline=mock_pipe)
        self.assertIsInstance(converter, SDInpaintBaldConverter)

    def test_empty_hair_mask_returns_safe_original(self):
        cfg = BaldConverterConfig()
        empty_mask = Image.new("L", (200, 200), color=0)
        converter = SDInpaintBaldConverter(cfg, pipeline=MagicMock())
        result = converter.convert(self.target_img, empty_mask)
        self.assertEqual(result.metadata.get("status"), "skipped_empty_mask")
        self.assertEqual(np.max(np.abs(np.asarray(result.bald_base) - np.asarray(self.target_img))), 0)


if __name__ == "__main__":
    unittest.main()
