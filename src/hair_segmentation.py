
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort


MODEL_PATH = Path("models/resnet18.onnx")
IMAGE_PATH = Path("assets/image.png")
OUTPUT_DIR = Path("outputs")

HAIR_CLASS = 17
INPUT_SIZE = (512, 512)
#hey there !

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the original image
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {IMAGE_PATH.resolve()}"
        )

    original_height, original_width = image.shape[:2]

    # Convert BGR to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize while preserving the aspect ratio
    target_w, target_h = INPUT_SIZE
    h, w = rgb.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        rgb,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # Pad to 512 x 512 without stretching the image
    canvas = np.zeros(
        (target_h, target_w, 3),
        dtype=np.uint8,
    )

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w,
    ] = resized

    # Normalize the padded image using ImageNet statistics
    x = canvas.astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    ).reshape(1, 1, 3)

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    ).reshape(1, 1, 3)

    x = (x - mean) / std

    # HWC -> CHW -> NCHW
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    x = np.ascontiguousarray(x, dtype=np.float32)

    # Initialize CPU inference
    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    print("Input image:", original_width, "x", original_height)
    print("Running segmentation on CPU...")

    start = time.perf_counter()

    predictions = session.run(
        None,
        {session.get_inputs()[0].name: x},
    )

    elapsed = time.perf_counter() - start

    # Inspect all outputs
    for output_info, output in zip(session.get_outputs(), predictions):
        print(
            f"Output {output_info.name}: "
            f"shape={output.shape}, "
            f"min={output.min():.3f}, "
            f"max={output.max():.3f}"
        )

    # Use the primary output as the segmentation logits.
    # We will validate this output against the model's documentation.
    for output_info, output in zip(session.get_outputs(), predictions):
        segmentation_i = np.argmax(output, axis=1)[0]
        mask_i = (segmentation_i == HAIR_CLASS)
        # Remove padding and restore original dimensions
        mask_i = mask_i[
            y_offset:y_offset + new_h,
            x_offset:x_offset + new_w,
        ]
        mask_i = cv2.resize(
            mask_i.astype(np.uint8),
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
        coverage_i = 100.0 * np.count_nonzero(mask_i) / mask_i.size
        print(
            f"{output_info.name}: "
            f"hair pixels={np.count_nonzero(mask_i)}, "
            f"coverage={coverage_i:.2f}%"
        )
    logits = predictions[0]

    # Choose the highest-scoring class for every pixel
    segmentation = np.argmax(logits, axis=1)[0].astype(np.uint8)

    # Create the binary hair mask (class 17)
    hair_mask_512 = (
        segmentation == HAIR_CLASS
    ).astype(np.uint8) * 255

    # Remove padding before restoring the original image size
    hair_mask_resized = hair_mask_512[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w,
    ]

    hair_mask = cv2.resize(
        hair_mask_resized,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )

    # Highlight detected hair in green
    overlay = image.copy()
    overlay[hair_mask > 0] = (0, 255, 0)

    blended = cv2.addWeighted(
        image,
        0.65,
        overlay,
        0.35,
        0,
    )

    # Save the results
    mask_path = OUTPUT_DIR / "hair_mask.png"
    overlay_path = OUTPUT_DIR / "hair_overlay.png"

    cv2.imwrite(str(mask_path), hair_mask)
    cv2.imwrite(str(overlay_path), blended)

    total_pixels = hair_mask.size
    hair_pixels = int(np.count_nonzero(hair_mask))
    coverage = 100.0 * hair_pixels / total_pixels

    print("\nSegmentation inference completed.")
    print(f"Inference time: {elapsed:.2f} seconds")
    print(f"Hair pixels: {hair_pixels}")
    print(f"Hair coverage: {coverage:.2f}%")
    print(f"Mask saved to: {mask_path.resolve()}")
    print(f"Overlay saved to: {overlay_path.resolve()}")


if __name__ == "__main__":
    main()
