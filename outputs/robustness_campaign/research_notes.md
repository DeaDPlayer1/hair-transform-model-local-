# HairTryOn Robustness Campaign — Research Notes

## Technical References & Investigated Solutions

### 1. Hair Mask Cleanup & False-Positive Pruning
- **Topic**: Removing studio/backdrop/shoulder false positives from reference segmentation masks.
- **Reference**: Official OpenCV Connected Components & Morphological Filtering (`cv2.connectedComponentsWithStats`, `cv2.morphologyEx`).
- **Application**: Reference masks frequently segment dark or textured backdrops behind the head. Pruning isolated connected components that do not overlap or connect to the central head bounding box removes background contamination before alignment.

### 2. Multi-Stage Envelopes and Restoration Safety
- **Topic**: Outside-envelope pixel restoration in chained diffusion pipelines.
- **Reference**: Stable Diffusion Inpainting compositing standards (Hugging Face Diffusers Pipeline design).
- **Application**: In a two-stage pipeline where Stage 1 removes hair and Stage 2 adds a new hairstyle, restoring original pixels outside the edit region requires:
  $$\text{Full Edit Envelope} = \text{Stage 1 Edit Mask} \cup \text{Stage 2 Broad Edit Mask}$$
  Restoring original pixels with only $\text{Stage 2 Mask}$ copies the target's original hair back into any region where Stage 1 shaved the hair but Stage 2 did not place new hair (e.g., when transitioning from long target hair to short reference hair).

### 3. Identity and Feature Protection
- **Topic**: Ear and temple boundary preservation during facial inpainting.
- **Reference**: MediaPipe Face Mesh landmark topology (478 landmarks) and facial contour isolation.
- **Application**: Protecting ears and eyes with a 0-tolerance hard boundary while maintaining a narrow, feather-dilated seam for hair strands ensures natural blending without blurring ear cartilage or eyebrows.
