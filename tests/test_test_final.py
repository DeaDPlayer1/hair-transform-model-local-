"""
Unit tests for test_final.py image matching and CLI wrapper.
"""

import unittest
from pathlib import Path
from test_final import match_image, scan_assets, normalize_token


class TestFinalRunner(unittest.TestCase):
    def test_normalize_token(self):
        self.assertEqual(normalize_token("Target 2.jpg"), "target2jpg")
        self.assertEqual(normalize_token("reference_4"), "reference4")
        self.assertEqual(normalize_token("hair-reference"), "hairreference")

    def test_scan_assets(self):
        assets = scan_assets(Path("assets"))
        self.assertGreaterEqual(len(assets["targets"]), 5)
        self.assertGreaterEqual(len(assets["references"]), 5)

    def test_match_image_exact_and_normalized(self):
        assets_dir = Path("assets")
        self.assertEqual(match_image("target.jpg", assets_dir).name, "target.jpg")
        self.assertEqual(match_image("target2", assets_dir).name, "target2.jpg")
        self.assertEqual(match_image("target 2.jpg", assets_dir).name, "target2.jpg")
        self.assertEqual(match_image("reference4", assets_dir).name, "reference4.jpg")
        self.assertEqual(match_image("reference_4.jpg", assets_dir).name, "reference4.jpg")
        self.assertEqual(match_image("target1", assets_dir).name, "target1.png")


if __name__ == "__main__":
    unittest.main()
