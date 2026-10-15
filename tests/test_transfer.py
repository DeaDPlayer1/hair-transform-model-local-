"""
Integration tests for hairstyle transfer pipeline.
"""
from pathlib import Path
import unittest
import numpy as np
from PIL import Image

from src.hair_transfer import transfer_hairstyle
from src.face_landmarks import detect_face_landmarks, analyze_pose_and_mirroring
import cv2


class TestHairstyleTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target_img = Image.open("assets/download.jpg").convert("RGB")
        cls.ref_img = Image.open("assets/image.png").convert("RGB")

        # Create dummy hair masks for testing
        w_t, h_t = cls.target_img.size
        w_r, h_r = cls.ref_img.size

        # Upper 30% hair dummy mask
        t_mask = np.zeros((h_t, w_t), dtype=np.uint8)
        t_mask[: int(h_t * 0.35), int(w_t * 0.2) : int(w_t * 0.8)] = 255
        cls.target_mask = Image.fromarray(t_mask, mode="L")

        r_mask = np.zeros((h_r, w_r), dtype=np.uint8)
        r_mask[: int(h_r * 0.5), :] = 255
        cls.ref_mask = Image.fromarray(r_mask, mode="L")

    def test_landmark_detection(self):
        tgt_bgr = cv2.cvtColor(np.array(self.target_img), cv2.COLOR_RGB2BGR)
        res = detect_face_landmarks(tgt_bgr)
        self.assertTrue(res["detected"])
        self.assertEqual(len(res["landmarks_2d"]), 478)
        self.assertAlmostEqual(res["yaw"], 0.8, delta=10.0)

    def test_pose_and_mirroring_analysis(self):
        tgt_bgr = cv2.cvtColor(np.array(self.target_img), cv2.COLOR_RGB2BGR)
        ref_bgr = cv2.cvtColor(np.array(self.ref_img), cv2.COLOR_RGB2BGR)

        tgt_lms = detect_face_landmarks(tgt_bgr)
        ref_lms = detect_face_landmarks(ref_bgr)

        pose = analyze_pose_and_mirroring(tgt_lms, ref_lms)
        self.assertIn("target_yaw", pose)
        self.assertIn("reference_yaw", pose)
        self.assertIn("recommend_mirror", pose)

    def test_transfer_dimensions(self):
        out_img, diag = transfer_hairstyle(
            self.target_img,
            self.ref_img,
            self.target_mask,
            self.ref_mask,
            method="landmark_affine",
            return_diagnostics=True,
        )
        self.assertEqual(out_img.size, self.target_img.size)
        self.assertIn("yaw_mismatch", diag)
        self.assertIn("alignment_mode", diag)

    def test_mirror_vs_unmirrored(self):
        out_mirrored, diag_m = transfer_hairstyle(
            self.target_img,
            self.ref_img,
            self.target_mask,
            self.ref_mask,
            mirror_reference=True,
            return_diagnostics=True,
        )
        out_unmirrored, diag_u = transfer_hairstyle(
            self.target_img,
            self.ref_img,
            self.target_mask,
            self.ref_mask,
            mirror_reference=False,
            return_diagnostics=True,
        )
        self.assertEqual(out_mirrored.size, self.target_img.size)
        self.assertEqual(out_unmirrored.size, self.target_img.size)
        self.assertTrue(diag_m["is_mirrored"])
        self.assertFalse(diag_u["is_mirrored"])

    def test_baseline_bbox_fallback(self):
        out_base, diag_b = transfer_hairstyle(
            self.target_img,
            self.ref_img,
            self.target_mask,
            self.ref_mask,
            method="baseline_bbox",
            return_diagnostics=True,
        )
        self.assertEqual(out_base.size, self.target_img.size)
        self.assertEqual(diag_b["alignment_mode"], "baseline_bbox")


if __name__ == "__main__":
    unittest.main()
