"""
Face landmark detection, 3D head pose estimation, hairline estimation,
and anatomical head envelope constraints using MediaPipe CPU Face Landmarker.
"""

from pathlib import Path
import urllib.request
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_TASK_PATH = Path("models/face_landmarker.task")


def ensure_task_model(model_path: Path = DEFAULT_TASK_PATH) -> Path:
    """Ensure the MediaPipe face_landmarker.task model is present on disk."""
    model_path = Path(model_path)
    if not model_path.is_file():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face landmarker model to {model_path}...")
        urllib.request.urlretrieve(TASK_MODEL_URL, model_path)
    return model_path


# Landmark indices for midline and symmetric pairs:
MIDLINE_LANDMARK_INDICES = [10, 151, 9, 8, 168, 6, 1, 152]

# Bilateral symmetric pairs in MediaPipe Face Mesh:
# (viewer-left feature <-> viewer-right feature)
BILATERAL_SYMMETRIC_PAIRS = {
    33: 263, 263: 33,    # Outer eye corners
    133: 362, 362: 133,  # Inner eye corners
    70: 300, 300: 70,    # Outer eyebrow
    105: 334, 334: 105,  # Mid eyebrow
    63: 296, 296: 63,    # Inner eyebrow
    234: 454, 454: 234,  # Tragus / ear / temple
    127: 356, 356: 127,  # Cheek boundary
    109: 338, 338: 109,  # Forehead boundary
    103: 332, 332: 103,  # Temple boundary
}

ANCHOR_LANDMARK_INDICES = [
    10, 151, 9, 8, 168, 6, 1,
    33, 133, 362, 263,
    70, 105, 300, 334,
    234, 454,
]


def detect_face_landmarks(
    image_bgr: np.ndarray,
    model_path: Path = DEFAULT_TASK_PATH,
    min_confidence: float = 0.1,
):
    """
    Detect 478 face landmarks and estimate 3D head pose on CPU.

    Returns:
        dict containing:
            - 'detected': bool
            - 'landmarks_2d': np.ndarray of shape (478, 2) in pixel coordinates, or None
            - 'landmarks_3d': np.ndarray of shape (478, 3) in pixel scale, or None
            - 'yaw': float head rotation around vertical axis in degrees (-90 to +90)
            - 'pitch': float head pitch in degrees
            - 'roll': float head roll / tilt in degrees
    """
    ensure_task_model(model_path)

    base_opts = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    opts = vision.FaceLandmarkerOptions(
        base_options=base_opts,
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )

    try:
        landmarker = vision.FaceLandmarker.create_from_options(opts)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
    except Exception as exc:
        print(f"MediaPipe detection failed: {exc}")
        return {
            "detected": False,
            "landmarks_2d": None,
            "landmarks_3d": None,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
        }

    if not result.face_landmarks:
        return {
            "detected": False,
            "landmarks_2d": None,
            "landmarks_3d": None,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
        }

    raw_lms = result.face_landmarks[0]
    height, width = image_bgr.shape[:2]

    lms_2d = np.array(
        [(pt.x * width, pt.y * height) for pt in raw_lms],
        dtype=np.float32,
    )
    lms_3d = np.array(
        [(pt.x * width, pt.y * height, pt.z * width) for pt in raw_lms],
        dtype=np.float32,
    )

    yaw, pitch, roll = 0.0, 0.0, 0.0
    if result.facial_transformation_matrixes:
        mat = np.array(result.facial_transformation_matrixes[0])
        R = mat[:3, :3]
        yaw = float(np.arctan2(R[0, 2], R[2, 2]) * 180.0 / np.pi)
        pitch = float(
            np.arctan2(-R[1, 2], np.sqrt(R[0, 2] ** 2 + R[2, 2] ** 2))
            * 180.0
            / np.pi
        )
        roll = float(np.arctan2(R[1, 0], R[1, 1]) * 180.0 / np.pi)
    else:
        left_eye = lms_2d[33]
        right_eye = lms_2d[263]
        nose = lms_2d[1]
        eye_width = max(1.0, right_eye[0] - left_eye[0])
        rel_pos = (nose[0] - left_eye[0]) / eye_width
        yaw = float((rel_pos - 0.5) * 100.0)

    return {
        "detected": True,
        "landmarks_2d": lms_2d,
        "landmarks_3d": lms_3d,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
    }


def get_mirrored_landmarks(landmarks_2d: np.ndarray, image_width: int) -> np.ndarray:
    """
    Consistently transform landmark coordinates to horizontally mirrored space.
    Flips the x coordinate: x_mirrored = (image_width - 1) - x.
    Preserves exact 1-to-1 pixel correspondence with cv2.flip(image, 1).
    """
    if landmarks_2d is None:
        return None
    mirrored = landmarks_2d.copy()
    mirrored[:, 0] = (image_width - 1) - mirrored[:, 0]
    return mirrored


def build_symmetric_correspondence(
    src_lms_2d: np.ndarray,
    dst_lms_2d: np.ndarray,
    is_src_mirrored: bool,
    anchor_indices=ANCHOR_LANDMARK_INDICES,
):
    """
    Build corresponding source and destination landmark point arrays.
    If the source is mirrored, swaps bilateral landmark indices so that
    anatomical features (left eye, right eye, left brow, etc.) map to the
    corresponding target features.
    """
    src_pts = []
    dst_pts = []

    for idx in anchor_indices:
        src_pt = src_lms_2d[idx]
        if is_src_mirrored:
            # Map source mirrored point to target symmetric feature index
            tgt_idx = BILATERAL_SYMMETRIC_PAIRS.get(idx, idx)
        else:
            tgt_idx = idx

        dst_pt = dst_lms_2d[tgt_idx]
        src_pts.append(src_pt)
        dst_pts.append(dst_pt)

    return np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32)


def analyze_pose_and_mirroring(target_info: dict, reference_info: dict):
    """
    Compare target and reference head poses and determine optimal horizontal mirroring.
    """
    t_yaw = target_info.get("yaw", 0.0) if target_info.get("detected") else 0.0
    r_yaw = reference_info.get("yaw", 0.0) if reference_info.get("detected") else 0.0

    unmirrored_diff = abs(t_yaw - r_yaw)
    mirrored_r_yaw = -r_yaw
    mirrored_diff = abs(t_yaw - mirrored_r_yaw)

    recommend_mirror = mirrored_diff < unmirrored_diff
    min_diff = min(unmirrored_diff, mirrored_diff)

    pose_warning = None
    if min_diff > 45.0:
        pose_warning = (
            f"Significant viewpoint mismatch ({min_diff:.1f}° difference). "
            f"Target face orientation is {t_yaw:+.1f}°, whereas reference hairstyle is {r_yaw:+.1f}°. "
            "2D compositing across large angle differences (e.g. frontal vs side-profile) "
            "cannot synthesize occluded 3D hair geometry."
        )
    elif min_diff > 25.0:
        pose_warning = (
            f"Moderate viewpoint difference ({min_diff:.1f}°). "
            "Aligned using facial anchors, but hair perspective will reflect the donor viewing angle."
        )

    return {
        "target_yaw": t_yaw,
        "reference_yaw": r_yaw,
        "unmirrored_diff": unmirrored_diff,
        "mirrored_diff": mirrored_diff,
        "recommend_mirror": recommend_mirror,
        "pose_mismatch_warning": pose_warning,
    }


def create_head_constraints(
    image_shape: tuple,
    target_landmarks: dict,
    segmentation_map: np.ndarray = None,
    target_hair_mask: np.ndarray = None,
):
    """
    Construct:
        1. permitted_envelope: binary mask (uint8, 255 = allowed) defining the
           plausible head and hair region. Strictly clips out clothing, lower neck,
           shoulders, and excessive lateral spill.
        2. protected_face_mask: binary mask (uint8, 255 = protected) covering
           eyes, eyebrows, nose, mouth, ears, and lower facial skin.

    Returns:
        (permitted_envelope, protected_face_mask)
    """
    height, width = image_shape[:2]
    permitted = np.ones((height, width), dtype=np.uint8) * 255
    protected = np.zeros((height, width), dtype=np.uint8)

    # Resolution-adaptive dilation kernel size (1px for 150px, 3px for 500px, 5px for 1000px)
    k_size = max(1, int(round(height * 0.004)))
    if k_size % 2 == 0:
        k_size += 1
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    # 1. Protect regions identified by segmentation model (CelebAMask classes)
    if segmentation_map is not None:
        # Classes: 2,3 (eyebrows), 4,5 (eyes), 6 (glasses), 7,8 (ears),
        # 10 (nose), 11,12,13 (mouth & lips)
        feature_classes = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
        feature_mask = np.isin(segmentation_map, feature_classes).astype(np.uint8) * 255
        feature_mask = cv2.dilate(feature_mask, dilate_kernel, iterations=1)
        protected = np.maximum(protected, feature_mask)

        # Strictly disallow hair on clothing (class 16)
        clothing_mask = (segmentation_map == 16).astype(np.uint8) * 255
        permitted[clothing_mask > 0] = 0

    # 2. Geometric head envelope from landmarks if available
    if target_landmarks and target_landmarks.get("detected"):
        lms = target_landmarks["landmarks_2d"]

        cx = float(lms[168, 0])  # Nasion center x
        cy = float(lms[168, 1])
        chin_y = float(lms[152, 1])

        ear_l_x = float(lms[234, 0])
        ear_r_x = float(lms[454, 0])
        ear_y = max(float(lms[234, 1]), float(lms[454, 1]))
        head_w = max(30.0, abs(ear_r_x - ear_l_x))

        # Vertical constraint: hair cannot extend below chin level
        # (unless the target already had long hair down there)
        chin_cutoff = int(chin_y + max(4, height * 0.02))
        if target_hair_mask is not None:
            # Preserve target hair region below chin if present
            below_chin = np.zeros((height, width), dtype=bool)
            below_chin[chin_cutoff:, :] = True
            invalid_below_chin = below_chin & (target_hair_mask == 0)
            permitted[invalid_below_chin] = 0
        else:
            permitted[chin_cutoff:, :] = 0

        # Lateral cranial envelope:
        for y in range(height):
            if y > ear_y:
                # Below ear level: tightly bound to jaw/neck
                allowed_hw = head_w * 0.75
                x1 = int(max(0, cx - allowed_hw))
                x2 = int(min(width, cx + allowed_hw))
                permitted[y, :x1] = 0
                permitted[y, x2:] = 0
            else:
                # Above ear level: allow cranial dome volume
                allowed_hw = head_w * 1.05
                x1 = int(max(0, cx - allowed_hw))
                x2 = int(min(width, cx + allowed_hw))
                permitted[y, :x1] = 0
                permitted[y, x2:] = 0

        # Protect face interior below eyebrows
        brow_y_min = min(
            float(lms[70, 1]), float(lms[63, 1]), float(lms[105, 1]),
            float(lms[336, 1]), float(lms[296, 1]), float(lms[334, 1])
        )
        # Eyebrow-to-chin facial polygon
        face_contour_pts = lms[[
            70, 63, 105, 66, 107,
            9, 8,
            336, 296, 334, 293, 300,
            356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
            148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 70
        ]].astype(np.int32)

        face_interior = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(face_interior, [face_contour_pts], 255)
        protected = np.maximum(protected, face_interior)

    # Hair can never overwrite protected facial features
    permitted[protected > 0] = 0

    return permitted, protected


def create_target_hairline_overlay(
    target_bgr: np.ndarray,
    target_landmarks: dict,
    segmentation_map: np.ndarray = None,
    permitted_envelope: np.ndarray = None,
) -> np.ndarray:
    """
    Create a diagnostic visualization overlay showing:
        - Target face boundary (jawline & cheeks) in green
        - Eyebrows in blue
        - Eyes in cyan
        - Ears & temples in magenta
        - Predicted hairline contour in yellow
        - Permitted head envelope boundary in orange
    """
    overlay = target_bgr.copy()
    h, w = target_bgr.shape[:2]
    line_thickness = max(1, int(round(h * 0.005)))
    pt_radius = max(2, int(round(h * 0.008)))

    if target_landmarks and target_landmarks.get("detected"):
        lms = target_landmarks["landmarks_2d"]

        # 1. Jawline & cheek boundary (Green)
        jaw_indices = [
            234, 127, 162, 21, 54, 103, 67, 109,
            10,
            338, 297, 332, 284, 251, 389, 356, 454,
            323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
            148, 176, 149, 150, 136, 172, 58, 132, 93, 234
        ]
        pts_jaw = lms[jaw_indices].astype(np.int32)
        cv2.polylines(overlay, [pts_jaw], True, (0, 255, 0), line_thickness)

        # 2. Eyebrows (Blue)
        left_brow = lms[[70, 63, 105, 66, 107]].astype(np.int32)
        right_brow = lms[[336, 296, 334, 293, 300]].astype(np.int32)
        cv2.polylines(overlay, [left_brow], False, (255, 128, 0), line_thickness + 1)
        cv2.polylines(overlay, [right_brow], False, (255, 128, 0), line_thickness + 1)

        # 3. Eyes (Cyan)
        left_eye = lms[[33, 160, 158, 133, 153, 144, 33]].astype(np.int32)
        right_eye = lms[[362, 385, 387, 263, 373, 380, 362]].astype(np.int32)
        cv2.polylines(overlay, [left_eye], True, (255, 255, 0), line_thickness)
        cv2.polylines(overlay, [right_eye], True, (255, 255, 0), line_thickness)

        # 4. Forehead & Predicted hairline (Yellow)
        hairline_pts = lms[[234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454]].astype(np.int32)
        cv2.polylines(overlay, [hairline_pts], False, (0, 255, 255), line_thickness + 1)

        # 5. Temples & Ears (Magenta)
        for e_idx in [234, 454]:
            pt = tuple(lms[e_idx].astype(np.int32))
            cv2.circle(overlay, pt, pt_radius + 2, (255, 0, 255), -1)

    # 6. Permitted head envelope contour (Orange)
    if permitted_envelope is not None:
        contours, _ = cv2.findContours(
            permitted_envelope, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 165, 255), line_thickness)

    return overlay
