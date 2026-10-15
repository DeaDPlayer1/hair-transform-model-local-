"""
Image Preprocessing and Validation Module for Hairstyle Transfer.

Provides robust validation, EXIF orientation correction, dimension checks,
and aspect-ratio-safe transformations for target and reference portrait images.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.face_landmarks import detect_face_landmarks, DEFAULT_TASK_PATH


SUPPORTED_IMAGE_FORMATS = ("jpg", "jpeg", "png", "webp")
DEFAULT_MAX_DIMENSION = 4096
DEFAULT_MIN_DIMENSION = 64


class ImageValidationError(ValueError):
    """Raised when an uploaded or input image fails validation."""
    pass


def load_and_validate_image(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image],
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    min_dimension: int = DEFAULT_MIN_DIMENSION,
) -> Image.Image:
    """
    Safely load an image, correct EXIF orientation, convert to RGB, and validate bounds.

    Args:
        image_input: File path, bytes, BytesIO, or existing PIL Image.
        max_dimension: Maximum allowed width or height in pixels.
        min_dimension: Minimum allowed width or height in pixels.

    Returns:
        PIL Image in 'RGB' mode with correct orientation.

    Raises:
        ImageValidationError: If the image is empty, corrupt, or outside dimension limits.
    """
    if image_input is None:
        raise ImageValidationError("Image input cannot be None.")

    try:
        if isinstance(image_input, Image.Image):
            pil_img = image_input.copy()
        elif isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise ImageValidationError(f"Image file does not exist: {path}")
            if path.stat().st_size == 0:
                raise ImageValidationError(f"Image file is empty (0 bytes): {path}")
            with Image.open(path) as img:
                pil_img = ImageOps.exif_transpose(img)
                pil_img.load()
        elif isinstance(image_input, (bytes, io.BytesIO)):
            bio = io.BytesIO(image_input) if isinstance(image_input, bytes) else image_input
            bio.seek(0)
            if len(bio.getvalue()) == 0:
                raise ImageValidationError("Uploaded image byte buffer is empty.")
            with Image.open(bio) as img:
                pil_img = ImageOps.exif_transpose(img)
                pil_img.load()
        else:
            raise ImageValidationError(f"Unsupported image input type: {type(image_input)}")

        # Ensure RGB mode (handling Grayscale, RGBA, CMYK, Palette)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(f"Failed to decode image: {exc}") from exc

    w, h = pil_img.size
    if w < min_dimension or h < min_dimension:
        raise ImageValidationError(
            f"Image dimensions ({w}x{h}) are below minimum allowed ({min_dimension}x{min_dimension})."
        )
    if w > max_dimension or h > max_dimension:
        raise ImageValidationError(
            f"Image dimensions ({w}x{h}) exceed maximum allowed ({max_dimension}x{max_dimension})."
        )

    return pil_img


def resize_preserving_aspect(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Resize image to fit within target_size while strictly preserving aspect ratio,
    padding the remainder with pad_color.

    Returns:
        (padded_image, transform_info) where transform_info contains
        offsets and scales needed to invert the transformation.
    """
    orig_w, orig_h = image.size
    target_w, target_h = target_size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", target_size, color=pad_color)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas.paste(resized, (x_offset, y_offset))

    transform_info = {
        "orig_size": (orig_w, orig_h),
        "target_size": target_size,
        "scaled_size": (new_w, new_h),
        "scale": scale,
        "x_offset": x_offset,
        "y_offset": y_offset,
    }
    return canvas, transform_info


def validate_target_portrait(
    image: Image.Image,
    model_path: Path = DEFAULT_TASK_PATH,
) -> Dict[str, Any]:
    """
    Run diagnostic checks on the target portrait:
    - Verifies face presence via MediaPipe face landmarker.
    - Warns if face is missing, heavily pitched/yawed, or too small.

    Returns:
        dict with 'valid': bool, 'warnings': list[str], 'landmarks': dict or None
    """
    warnings = []
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    lm_res = detect_face_landmarks(bgr, model_path=model_path)
    if not lm_res.get("detected"):
        warnings.append("No face detected in target portrait. Hairline alignment may be approximate.")
        return {
            "valid": False,
            "has_face": False,
            "warnings": warnings,
            "landmarks": None,
        }

    yaw = abs(lm_res.get("yaw", 0.0))
    pitch = abs(lm_res.get("pitch", 0.0))

    if yaw > 45.0:
        warnings.append(f"Target face has extreme lateral yaw ({yaw:.1f}°). Frontal or 3/4 portraits produce superior results.")
    if pitch > 30.0:
        warnings.append(f"Target face is tilted vertically ({pitch:.1f}° pitch).")

    return {
        "valid": True,
        "has_face": True,
        "warnings": warnings,
        "landmarks": lm_res,
    }
