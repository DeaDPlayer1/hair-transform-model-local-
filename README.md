# HairTryOn: Local Generative Hairstyle Transfer Engine

HairTryOn is a local, privacy-first generative hairstyle transfer application. It combines pretrained diffusion models with geometry-aware landmark alignment and semantic face parsing to seamlessly transfer donor hairstyles onto target portrait photos while preserving 100% of facial identity, skin tone, and unedited background pixels.

---

## 🌟 Key Architecture & Features

1. **Base Diffusion Inpainting**: Stable Diffusion v1.5 Inpainting (`runwayml/stable-diffusion-inpainting`).
2. **Identity & Texture Conditioning**: IP-Adapter Plus (`h94/IP-Adapter` with OpenCLIP ViT-H/14 image encoder).
3. **Geometry-Aware Alignment**:
   - MediaPipe Face Mesh landmark detection (478 3D landmarks).
   - Procrustes similarity transformation for scaling, rotation, and translation matching.
   - Cranial silhouette expansion and hairline transition zoning.
4. **Edge Decontamination & Antialiasing**:
   - Boundary-restricted inpainting ring to remove donor background contamination without smearing curls.
   - Signed-distance antialiased crop conditioning for IP-Adapter to prevent gray fringe artifacts.
5. **Facial Feature Protection**:
   - ResNet18 ONNX face parser segments eyes, eyebrows, nose, lips, mouth, neck, and ears to ensure the edit mask never erodes facial features.
6. **Feathered Seam Compositing**:
   - Targeted inpainting along boundary seams, guaranteeing bit-exact preservation of original face and background pixels.
7. **Interactive UI**:
   - Streamlit application (`app.py`) with alignment previews, generative parameter controls, and export options.

---

## 📁 Repository Structure

```
HairTryOn/
├── app.py                      # Streamlit interactive application
├── requirements.txt            # Python dependencies
├── assets/                     # Example test portraits & donor hairstyles
├── models/
│   ├── resnet18.onnx           # ResNet18 face parser ONNX model
│   └── face_landmarker.task    # MediaPipe Face Landmarker model
├── src/
│   ├── local_generative_transfer.py # Main diffusion transfer pipeline
│   ├── hairstyle_alignment.py       # Landmark-based geometric warping & decontam
│   ├── hair_segmentation.py         # Face parser hair segmentation
│   ├── hair_mask_refinement.py      # Edit mask & facial protection logic
│   ├── hair_transfer.py             # Classical 2D affine baseline
│   ├── face_landmarks.py            # MediaPipe facial landmark detector
│   ├── image_preprocessing.py       # Preprocessing & normalization utils
│   └── transfer_evaluation.py       # Quantitative metrics (PSNR, SSIM, mask coverage)
├── scripts/
│   ├── run_ablation_experiments.py  # 4-condition ablation benchmark runner
│   └── test_local_generative_transfer.py # CLI test runner
└── tests/                           # Pytest verification suite
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/DeaDPlayer1/hair-transform-model-local-.git
cd hair-transform-model-local-
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 3. Run Verification Tests

```bash
pytest -v
```

---

## 🔬 Ablation Experiments

To evaluate the edge decontamination and reference crop antialiasing fixes across 4 controlled conditions:

```bash
python scripts/run_ablation_experiments.py
```
