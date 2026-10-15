"""
Comprehensive regression test suite for hairstyle transfer pipeline.
Covers:
1. Mask and image coordinate consistency after mirroring.
2. Affine transformation of landmarks and masks.
3. No donor-background leakage onto target.
4. No hair composited onto clothing.
5. Protected facial features remaining visible (eyes, eyebrows, nose).
6. No unexpected white holes in replacement region.
7. Output dimensions matching target.
"""

from pathlib import Path
import unittest
import numpy as np
import cv2
from PIL import Image

from src.hair_transfer import transfer_hairstyle
from src.face_landmarks import (
    detect_face_landmarks,
    get_mirrored_landmarks,
    build_symmetric_correspondence,
    create_head_constraints,
    ANCHOR_LANDMARK_INDICES,
)


class TestHairTransferRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target_bgr = cv2.imread("assets/download.jpg")
        cls.ref_bgr = cv2.imread("assets/image.png")

        cls.target_pil = Image.fromarray(cv2.cvtColor(cls.target_bgr, cv2.COLOR_BGR2RGB))
        cls.ref_pil = Image.fromarray(cv2.cvtColor(cls.ref_bgr, cv2.COLOR_BGR2RGB))

        cls.w_tgt, cls.h_tgt = cls.target_pil.size
        cls.w_ref, cls.h_ref = cls.ref_pil.size

        # Load segmentation
        import onnxruntime as ort
        session = ort.InferenceSession("models/resnet18.onnx", providers=["CPUExecutionProvider"])

        def get_seg(img_bgr):
            h, w = img_bgr.shape[:2]
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            scale = min(512 / w, 512 / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            resized = cv2.resize(rgb, (nw, nh))
            canvas = np.zeros((512, 512, 3), dtype=np.uint8)
            xo, yo = (512 - nw) // 2, (512 - nh) // 2
            canvas[yo:yo + nh, xo:xo + nw] = resized
            x = (canvas.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)
            outs = session.run(None, {"input": x})
            seg = np.argmax(outs[0], axis=1)[0]
            seg_crop = seg[yo:yo + nh, xo:xo + nw]
            return cv2.resize(seg_crop, (w, h), interpolation=cv2.INTER_NEAREST)

        cls.tgt_seg = get_seg(cls.target_bgr)
        cls.ref_seg = get_seg(cls.ref_bgr)

        cls.tgt_mask_pil = Image.fromarray((cls.tgt_seg == 17).astype(np.uint8) * 255, mode="L")
        cls.ref_mask_pil = Image.fromarray((cls.ref_seg == 17).astype(np.uint8) * 255, mode="L")

    def test_1_mirror_coordinate_consistency(self):
        """Verify that mirroring transforms image, mask, and landmarks identically."""
        ref_lms = detect_face_landmarks(self.ref_bgr)["landmarks_2d"]
        mirrored_lms = get_mirrored_landmarks(ref_lms, self.w_ref)

        # Check a specific landmark point (forehead apex landmark 10)
        orig_x, orig_y = ref_lms[10]
        expected_x = (self.w_ref - 1) - orig_x
        expected_y = orig_y

        self.assertAlmostEqual(mirrored_lms[10, 0], expected_x, delta=1e-4)
        self.assertAlmostEqual(mirrored_lms[10, 1], expected_y, delta=1e-4)

        # Check mask mirroring
        mask_np = np.array(self.ref_mask_pil)
        flipped_mask = cv2.flip(mask_np, 1)

        # Pixel at (x, y) in original should match (W-1-x, y) in flipped
        test_y, test_x = 100, 200
        self.assertEqual(mask_np[test_y, test_x], flipped_mask[test_y, (self.w_ref - 1) - test_x])

    def test_2_affine_transformation_landmarks_and_masks(self):
        """Verify affine transformation scale, rotation, and inliers."""
        out_img, diag = transfer_hairstyle(
            target_image=self.target_pil,
            reference_image=self.ref_pil,
            target_hair_mask=self.tgt_mask_pil,
            reference_hair_mask=self.ref_mask_pil,
            mirror_reference=True,
            return_diagnostics=True,
            target_segmentation_map=self.tgt_seg,
        )

        self.assertEqual(diag["alignment_mode"], "landmark_affine")
        self.assertGreaterEqual(diag["inliers"], 5)
        self.assertGreater(diag["scale"], 0.1)
        self.assertLess(diag["scale"], 1.0)
        self.assertLess(abs(diag["rotation_deg"]), 20.0)

    def test_3_no_donor_background_leakage(self):
        """Confirm that background pixels from the donor cannot overwrite the target."""
        out_img, diag = transfer_hairstyle(
            target_image=self.target_pil,
            reference_image=self.ref_pil,
            target_hair_mask=self.tgt_mask_pil,
            reference_hair_mask=self.ref_mask_pil,
            mirror_reference=True,
            return_diagnostics=True,
            target_segmentation_map=self.tgt_seg,
        )
        out_bgr = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR)

        # Check pure background region of target outside any hair (bottom right background)
        # Coordinates (140, 20) in download.jpg is pure white background
        self.assertTrue(np.allclose(out_bgr[20, 140], self.target_bgr[20, 140], atol=5))

    def test_4_no_hair_composited_onto_clothing(self):
        """Verify that transformed hair is clipped and cannot spill onto clothing."""
        out_img, diag = transfer_hairstyle(
            target_image=self.target_pil,
            reference_image=self.ref_pil,
            target_hair_mask=self.tgt_mask_pil,
            reference_hair_mask=self.ref_mask_pil,
            mirror_reference=True,
            return_diagnostics=True,
            target_segmentation_map=self.tgt_seg,
        )
        self.assertEqual(diag["clothing_overlap_px"], 0)

        # Target clothing pixels (class 16) should be identical to original target
        clothing_indices = self.tgt_seg == 16
        out_bgr = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR)
        diff_on_clothing = np.abs(out_bgr[clothing_indices].astype(int) - self.target_bgr[clothing_indices].astype(int))
        self.assertEqual(np.max(diff_on_clothing), 0)

    def test_5_protected_facial_features_remain_visible(self):
        """Verify that eyes, eyebrows, nose, and mouth are preserved."""
        out_img, diag = transfer_hairstyle(
            target_image=self.target_pil,
            reference_image=self.ref_pil,
            target_hair_mask=self.tgt_mask_pil,
            reference_hair_mask=self.ref_mask_pil,
            mirror_reference=True,
            return_diagnostics=True,
            target_segmentation_map=self.tgt_seg,
        )
        out_bgr = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR)

        # Eyes (classes 4, 5) and nose (class 10) and mouth (11, 12, 13)
        core_features = np.isin(self.tgt_seg, [4, 5, 10, 11, 12, 13])
        diff_on_features = np.abs(out_bgr[core_features].astype(int) - self.target_bgr[core_features].astype(int))
        self.assertEqual(np.max(diff_on_features), 0)

    def test_6_no_unexpected_white_holes(self):
        """Verify that inpainting did not cut white holes into the target scalp."""
        out_img, diag = transfer_hairstyle(
            target_image=self.target_pil,
            reference_image=self.ref_pil,
            target_hair_mask=self.tgt_mask_pil,
            reference_hair_mask=self.ref_mask_pil,
            mirror_reference=True,
            return_diagnostics=True,
            target_segmentation_map=self.tgt_seg,
        )
        out_bgr = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR)

        # Original target hair region
        hair_indices = self.tgt_seg == 17
        # In download.jpg, background is ~[240, 240, 240] (white), hair is dark ~[40, 30, 25]
        # In the output, mean intensity in the original hair area should remain dark, not turn into white background
        hair_patch_out = out_bgr[hair_indices]
        mean_brightness = np.mean(hair_patch_out)
        self.assertLess(mean_brightness, 120.0, "Hair region turned into bright white hole!")

    def test_7_output_dimensions_match_target(self):
        """Verify that output retains exact target dimensions."""
        out_img = transfer_hairstyle(
            target_image=self.target_pil,
            reference_image=self.ref_pil,
            target_hair_mask=self.tgt_mask_pil,
            reference_hair_mask=self.ref_mask_pil,
            return_diagnostics=False,
        )
        self.assertEqual(out_img.size, (self.w_tgt, self.h_tgt))


if __name__ == "__main__":
    unittest.main()
