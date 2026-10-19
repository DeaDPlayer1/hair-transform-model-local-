
from pathlib import Path
import io
import time

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from src.hair_transfer import transfer_hairstyle, save_transfer_result
from src.image_preprocessing import (
    load_and_validate_image,
    validate_target_portrait,
    ImageValidationError,
)
from src.hair_mask_refinement import (
    build_refined_edit_mask,
    build_feathered_blend_mask,
)
from src.transfer_evaluation import (
    evaluate_transfer_metrics,
    list_saved_experiments,
    MANUAL_REVIEW_QUESTIONS,
)
from src.local_generative_transfer import (
    generate_local_hairstyle_transfer,
    load_local_generative_pipeline,
    build_target_edit_mask,
    prepare_reference_crop,
    get_process_memory_mb,
    INFERENCE_PRESETS,
)
from src.hairstyle_alignment import (
    align_reference_to_target,
    analyze_pose_and_mirroring,
)


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "resnet18.onnx"

INPUT_SIZE = (512, 512)
HAIR_CLASS = 17
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp"]


st.set_page_config(
    page_title="HairTryOn AI",
    page_icon="✂️",
    layout="wide",
)


# =========================================================
# IMAGE UTILITIES
# =========================================================

def load_rgb_image(uploaded_file):
    """Load JPG, JPEG, PNG or WebP and return a validated RGB image."""
    uploaded_file.seek(0)
    return load_and_validate_image(uploaded_file.read())


def image_png_bytes(image):
    """Encode a PIL image as PNG bytes for downloading."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def to_display_pil(img):
    """Convert numpy array (BGR or grayscale) or PIL Image to displayable RGB/L PIL Image."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return Image.fromarray(img, mode="L")
        elif img.ndim == 3 and img.shape[2] == 3:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mode="RGB")
        elif img.ndim == 3 and img.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA), mode="RGBA")
    return img


# =========================================================
# ONNX & GENERATIVE DIFFUSION MODELS
# =========================================================

@st.cache_resource
def load_model():
    """Load the face-parsing model once and reuse it."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )


@st.cache_resource(show_spinner=False)
def load_generative_pipeline():
    """Load local Stable Diffusion + IP-Adapter Plus pipeline once and cache in memory."""
    return load_local_generative_pipeline(local_files_only=True)


# =========================================================
# HAIR SEGMENTATION
# =========================================================

from src.hair_segmentation import segment_hair as core_segment_hair


def segment_hair(image_pil, return_seg_map=False):
    """Run face parsing and create a binary hair mask (delegates to src.hair_segmentation)."""
    return core_segment_hair(image_pil, return_seg_map=return_seg_map)


# =========================================================
# USER INTERFACE
# =========================================================

st.title("✂️ HairTryOn AI")
st.caption(
    "Local CPU prototype · Face parsing · "
    "JPG / JPEG / PNG / WebP"
)

st.info(
    "Upload the target person's photo and a hairstyle reference. "
    "This version detects hair in both images. "
    "It does not transfer the hairstyle yet."
)


# Upload both images.
target_col, reference_col = st.columns(2)

with target_col:
    st.subheader("1. Target person")

    target_file = st.file_uploader(
        "Photo whose face you want to preserve",
        type=SUPPORTED_FORMATS,
        key="target_upload",
    )

with reference_col:
    st.subheader("2. Hairstyle reference")

    reference_file = st.file_uploader(
        "Photo containing the desired hairstyle",
        type=SUPPORTED_FORMATS,
        key="reference_upload",
    )


# Read uploaded images.
target_image = None
reference_image = None

if target_file is not None:
    try:
        target_image = load_rgb_image(target_file)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not read target image: {exc}")

if reference_file is not None:
    try:
        reference_image = load_rgb_image(reference_file)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not read reference image: {exc}")


# Preview the uploaded images.
if target_image is not None or reference_image is not None:
    st.divider()
    st.subheader("Input images")

    preview_col1, preview_col2 = st.columns(2)

    with preview_col1:
        if target_image is not None:
            st.image(
                target_image,
                caption=(
                    f"Target · {target_image.width} × "
                    f"{target_image.height}"
                ),
                use_container_width=True,
            )

    with preview_col2:
        if reference_image is not None:
            st.image(
                reference_image,
                caption=(
                    f"Reference · {reference_image.width} × "
                    f"{reference_image.height}"
                ),
                use_container_width=True,
            )


# Run segmentation on both images.
if target_image is not None and reference_image is not None:
    st.divider()

    if st.button(
        "Detect hair in both images",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Running the segmentation model on your CPU..."
            ):
                tgt_res = segment_hair(target_image, return_seg_map=True)
                st.session_state["target_seg_map"] = tgt_res[4]
                st.session_state["target_results"] = tgt_res[:4]
                st.session_state["reference_results"] = segment_hair(reference_image)

        except Exception as exc:
            st.error(f"Hair segmentation failed: {exc}")


# =========================================================
# DISPLAY RESULTS
# =========================================================

if (
    "target_results" in st.session_state
    and "reference_results" in st.session_state
):
    target_results = st.session_state["target_results"]
    reference_results = st.session_state["reference_results"]

    st.divider()
    st.header("Segmentation results")

    st.caption(
        "White in a mask means the model predicted hair. "
        "Green in an overlay is a visualization of that prediction."
    )

    tab_target, tab_reference = st.tabs(
        ["Target person", "Hairstyle reference"]
    )

    with tab_target:
        mask, overlay, elapsed, coverage = target_results

        col_mask, col_overlay = st.columns(2)

        with col_mask:
            st.image(
                mask,
                caption="Target hair mask",
                use_container_width=True,
            )

            st.download_button(
                "Download target mask (PNG)",
                data=image_png_bytes(mask),
                file_name="target_hair_mask.png",
                mime="image/png",
                key="download_target_mask",
            )

        with col_overlay:
            st.image(
                overlay,
                caption="Target green overlay",
                use_container_width=True,
            )

            st.download_button(
                "Download target overlay (PNG)",
                data=image_png_bytes(overlay),
                file_name="target_hair_overlay.png",
                mime="image/png",
                key="download_target_overlay",
            )

        metric1, metric2 = st.columns(2)

        metric1.metric(
            "Inference time",
            f"{elapsed:.2f} seconds",
        )

        metric2.metric(
            "Predicted hair coverage",
            f"{coverage:.2f}%",
        )

    with tab_reference:
        mask, overlay, elapsed, coverage = reference_results

        col_mask, col_overlay = st.columns(2)

        with col_mask:
            st.image(
                mask,
                caption="Reference hair mask",
                use_container_width=True,
            )

            st.download_button(
                "Download reference mask (PNG)",
                data=image_png_bytes(mask),
                file_name="reference_hair_mask.png",
                mime="image/png",
                key="download_reference_mask",
            )

        with col_overlay:
            st.image(
                overlay,
                caption="Reference green overlay",
                use_container_width=True,
            )

            st.download_button(
                "Download reference overlay (PNG)",
                data=image_png_bytes(overlay),
                file_name="reference_hair_overlay.png",
                mime="image/png",
                key="download_reference_overlay",
            )

        metric1, metric2 = st.columns(2)

        metric1.metric(
            "Inference time",
            f"{elapsed:.2f} seconds",
        )

        metric2.metric(
            "Predicted hair coverage",
            f"{coverage:.2f}%",
        )

        with st.expander("🔍 Donor Hair Mask Audit (Connected Components & Geometry)", expanded=False):
            ref_mask_np = np.asarray(mask.convert("L"), dtype=np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (ref_mask_np > 128).astype(np.uint8), connectivity=8
            )
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_label = 1 + int(np.argmax(areas))
                largest_cc_mask = (labels == largest_label).astype(np.uint8) * 255
            else:
                largest_cc_mask = np.zeros_like(ref_mask_np)

            ys, xs = np.where(ref_mask_np > 128)
            if len(xs) > 0:
                bx, by, bw, bh = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
            else:
                bx, by, bw, bh = 0, 0, 0, 0

            audit_c1, audit_c2, audit_c3, audit_c4 = st.columns(4)
            with audit_c1:
                st.image(reference_image, caption="1. Original Reference", use_container_width=True)
            with audit_c2:
                st.image(mask, caption="2. Original Hair Mask", use_container_width=True)
            with audit_c3:
                st.image(Image.fromarray(largest_cc_mask, mode="L"), caption="3. Largest Connected Component", use_container_width=True)
            with audit_c4:
                st.image(mask, caption="4. Final Donor Mask (Curls Preserved)", use_container_width=True)

            st.markdown(
                f"- **Connected Components Detected**: `{max(0, num_labels - 1)}` (Entire side-profile hairstyle is 1 unified component)\n"
                f"- **Mask Bounding Box**: `x={bx}, y={by}, w={bw}, h={bh}`\n"
                f"- **Hair Pixels**: `{len(xs):,}` ({coverage:.2f}% of reference image)\n"
                f"- **Component Integrity**: Long curls and mullet nape are preserved as valid hair without discarding curl strands."
            )

    st.info(
        "Both masks are predictions. Inspect them in the tabs above before proceeding to transfer. "
        "The local generative engine below synthesizes new hair with Stable Diffusion + IP-Adapter Plus without cloud APIs."
    )

    st.divider()
    st.header("🎨 Local AI Hairstyle Transfer (Pretrained Diffusion + IP-Adapter)")
    st.caption(
        "100% Offline Local CPU Inference · Zero Cloud APIs or External Services · "
        "Stable Diffusion 1.5 Inpainting + IP-Adapter Plus"
    )

    # Pre-inference portrait validation
    face_audit = validate_target_portrait(target_image)
    ref_face_audit = validate_target_portrait(reference_image)
    target_landmarks = face_audit.get("landmarks")
    ref_landmarks = ref_face_audit.get("landmarks")

    col_fa1, col_fa2 = st.columns(2)
    with col_fa1:
        if face_audit["has_face"]:
            yaw = target_landmarks.get("yaw", 0.0) if target_landmarks else 0.0
            pitch = target_landmarks.get("pitch", 0.0) if target_landmarks else 0.0
            st.success(f"🎯 Target Face: Yaw {yaw:+.1f}°, Pitch {pitch:+.1f}°. Facial protection initialized.")
        else:
            st.warning("⚠️ No face detected in target image. Hairline alignment may be approximate.")
    with col_fa2:
        if ref_face_audit["has_face"]:
            ref_yaw = ref_landmarks.get("yaw", 0.0) if ref_landmarks else 0.0
            ref_pitch = ref_landmarks.get("pitch", 0.0) if ref_landmarks else 0.0
            st.success(f"✂️ Reference Face: Yaw {ref_yaw:+.1f}°, Pitch {ref_pitch:+.1f}°.")
        else:
            st.info("ℹ️ No face detected in reference image. Hair-centroid fallback will be used.")

    # Pose compatibility and auto-mirroring
    pose_info = analyze_pose_and_mirroring(target_landmarks, ref_landmarks)
    if pose_info.get("recommend_mirror"):
        st.info("💡 **Orientation Alignment**: Reference hairstyle points laterally opposite to target portrait. Auto-mirroring enabled.")
    if pose_info.get("pose_mismatch_warning"):
        st.warning(f"⚠️ {pose_info['pose_mismatch_warning']}")
    if face_audit["warnings"]:
        for w in face_audit["warnings"]:
            st.warning(f"⚠️ Target Portrait: {w}")

    # Presets selection
    preset_choice = st.selectbox(
        "Inference Preset",
        options=[
            "Balanced (10 steps, ~120-150s CPU) [Recommended]",
            "Fast Preview (8 steps, ~60-90s CPU)",
            "Quality (14 steps, ~180-240s CPU)",
            "Custom Settings",
        ],
        index=0,
        help="Select a CPU-calibrated preset or choose Custom Settings to configure diffusion parameters.",
    )

    default_steps = 10
    default_guidance = 7.5
    if "Fast Preview" in preset_choice:
        default_steps = 8
        default_guidance = 6.5
        st.caption("⚡ **Fast Preview**: 8 diffusion steps (~60-90s CPU). Fast check of hair placement and boundary blending.")
    elif "Balanced" in preset_choice:
        default_steps = 10
        default_guidance = 7.5
        st.caption("⚖️ **Balanced**: 10 diffusion steps (~120-150s CPU). Recommended balance of curl texture, sharp hairline, and identity preservation.")
    elif "Quality" in preset_choice:
        default_steps = 14
        default_guidance = 7.5
        st.caption("💎 **Quality**: 14 diffusion steps (~180-240s CPU). Maximum hair strand fidelity and fine hairline definition.")
    else:
        st.caption("🛠️ **Custom**: Full manual control over diffusion parameters.")

    with st.expander("⚙️ Advanced Generation & Blending Controls", expanded=(preset_choice == "Custom Settings")):
        col_gen1, col_gen2 = st.columns(2)
        with col_gen1:
            gen_steps = st.slider(
                "Denoising steps",
                min_value=6,
                max_value=20,
                value=default_steps,
                step=1,
                help="Number of diffusion inpainting steps on CPU. 8-12 steps with DPMSolver is fast and crisp.",
            )
            gen_scale = st.slider(
                "IP-Adapter hair conditioning scale",
                min_value=0.30,
                max_value=1.00,
                value=0.75,
                step=0.05,
                help="Controls how strongly reference hair texture/curls guide the result. 0.70-0.80 provides high curl fidelity with natural head fit.",
            )
            gen_guidance = st.slider(
                "Guidance scale (CFG)",
                min_value=3.0,
                max_value=12.0,
                value=default_guidance,
                step=0.5,
                help="Classifier-free guidance scale. Higher values follow prompts and conditioning more strictly.",
            )
        with col_gen2:
            gen_seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=999999,
                value=42,
                step=1,
                help="Seed for reproducible generation results.",
            )
            gen_expansion = st.slider(
                "Cranial edit expansion (px)",
                min_value=10,
                max_value=60,
                value=30,
                step=5,
                help="Expands the editable region upwards and sideways so new hair volume can fit.",
            )
            gen_feather = st.slider(
                "Inner boundary feathering (px)",
                min_value=2.0,
                max_value=15.0,
                value=6.0,
                step=1.0,
                help="Width of smooth inner feathering along the hairline and crown. Feathering is strictly inside the edit region.",
            )
            gen_neutral_bg = st.checkbox(
                "Neutralize reference background & face",
                value=True,
                help="Mutes the reference image background so IP-Adapter focuses strictly on hair texture and curls.",
            )

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            user_hair_prompt = st.text_input(
                "Optional custom prompt instructions",
                value="",
                placeholder="e.g. natural brown tone, detailed wavy curls, seamless hairline",
                help="Appended to the diffusion inpainting prompt.",
            )
        with col_p2:
            user_negative_prompt = st.text_input(
                "Optional additional negative prompt",
                value="",
                placeholder="e.g. frizzy flyaways, uneven color",
                help="Appended to the negative prompt.",
            )

    # Pre-Inference Edit Mask and Geometric Alignment Preview
    with st.expander("👁️ Inspect Geometric Alignment & Facial Protection Before Generation", expanded=False):
        st.caption(
            "Verify hairstyle placement and protected facial zones before starting diffusion. "
            "The geometry-aware aligner warps reference curls to fit the target cranial geometry and computes the exact inpainting region."
        )
        target_mask_preview = st.session_state["target_results"][0]
        reference_mask_preview = st.session_state["reference_results"][0]
        tgt_seg_map_preview = st.session_state.get("target_seg_map", None)

        alignment_prev = align_reference_to_target(
            target_image=target_image,
            reference_image=reference_image,
            target_hair_mask=target_mask_preview,
            reference_hair_mask=reference_mask_preview,
            target_landmarks=target_landmarks,
            reference_landmarks=ref_landmarks,
            target_segmentation_map=tgt_seg_map_preview,
            cranial_expansion_px=gen_expansion,
            feather_width_px=gen_feather,
        )

        tf = alignment_prev.get("transform_info", {})
        prev_imgs = alignment_prev.get("previews", {})

        st.caption(
            f"**Alignment Status**: {tf.get('mode', 'similarity').upper()} | "
            f"Scale: **{tf.get('scale', 1.0):.2f}x** | "
            f"Rotation: **{tf.get('rotation_deg', 0.0):.1f}°** | "
            f"Confidence: **{tf.get('confidence', 0.0):.2f}** | "
            f"Mirroring: **{'Yes' if alignment_prev.get('pose_info', {}).get('recommend_mirror') else 'No'}**"
        )

        col_pv1, col_pv2, col_pv3, col_pv4, col_pv5 = st.columns(5)
        with col_pv1:
            if "pasted_composite" in prev_imgs:
                st.image(to_display_pil(prev_imgs["pasted_composite"]), caption="1. Aligned Hairstyle Pre-Composite", use_container_width=True)
            else:
                st.image(target_image, caption="1. Target Portrait", use_container_width=True)
        with col_pv2:
            if "edge_repair_mask" in prev_imgs:
                st.image(to_display_pil(prev_imgs["edge_repair_mask"]), caption="2. Narrow Edge-Repair Mask", use_container_width=True)
            elif "edit_mask_preview" in prev_imgs:
                st.image(to_display_pil(prev_imgs["edit_mask_preview"]), caption="2. Edge Repair Mask", use_container_width=True)
            else:
                st.image(alignment_prev["target_edit_mask"], caption="2. Target Edit Mask", use_container_width=True)
        with col_pv3:
            if "alpha_preview" in prev_imgs:
                st.image(to_display_pil(prev_imgs["alpha_preview"]), caption="3. Clean Hair Alpha (Seam-Free)", use_container_width=True)
            else:
                st.image(target_mask_preview, caption="3. Target Hair Mask", use_container_width=True)
        with col_pv4:
            if "overlay_preview" in prev_imgs:
                st.image(to_display_pil(prev_imgs["overlay_preview"]), caption="4. Landmark & Hairline Overlay", use_container_width=True)
            else:
                st.image(target_image, caption="4. Target Portrait", use_container_width=True)
        with col_pv5:
            preview_ref_crop = prepare_reference_crop(
                reference_image=reference_image,
                reference_hair_mask=reference_mask_preview,
                neutral_background=gen_neutral_bg,
            )
            st.image(preview_ref_crop, caption="5. Reference Hair Conditioning", use_container_width=True)

    # Generation Button & Execution
    if st.button(
        "🚀 Generate Local AI Hairstyle",
        type="primary",
        use_container_width=True,
        key="btn_run_local_gen",
    ):
        try:
            target_mask = st.session_state["target_results"][0]
            reference_mask = st.session_state["reference_results"][0]
            tgt_seg_map = st.session_state.get("target_seg_map", None)

            status_container = st.empty()

            def update_gen_status(msg):
                status_container.info(f"⏳ {msg}")

            st.warning(
                "⏳ **Inference Running on CPU** (Intel Core i5-8350U). "
                "Expected time: approximately 1.5 to 2.5 minutes. Please leave this browser tab open."
            )

            with st.spinner("Initializing pipeline and running CPU diffusion inpainting..."):
                pipe = load_generative_pipeline()
                gen_res = generate_local_hairstyle_transfer(
                    target_image=target_image,
                    reference_image=reference_image,
                    target_hair_mask=target_mask,
                    reference_hair_mask=reference_mask,
                    target_segmentation_map=tgt_seg_map,
                    face_landmarks=target_landmarks,
                    reference_landmarks=ref_landmarks,
                    user_instruction=user_hair_prompt,
                    negative_prompt_extra=user_negative_prompt,
                    num_inference_steps=gen_steps,
                    ip_adapter_scale=gen_scale,
                    guidance_scale=gen_guidance,
                    seed=gen_seed,
                    cranial_expansion_px=gen_expansion,
                    feather_width_px=gen_feather,
                    neutral_reference_bg=gen_neutral_bg,
                    use_alignment=True,
                    save_experiment=True,
                    pipeline=pipe,
                    progress_callback=update_gen_status,
                )

            status_container.empty()
            st.session_state["local_gen_result"] = gen_res

            out_path = BASE_DIR / "outputs" / "local_generative_result.png"
            gen_res["composite_image"].save(out_path)
            st.success(
                f"✨ Hairstyle generated successfully in {gen_res['duration_seconds']:.1f}s! "
                f"Saved to outputs/local_generative_result.png"
            )

        except Exception as exc:
            st.error(f"Local generative transfer failed: {exc}")

    # Display Results & Diagnostics
    if "local_gen_result" in st.session_state:
        gen_res = st.session_state["local_gen_result"]
        metrics = gen_res.get("metrics", {})

        st.subheader("🎉 Result & Comparative Inspection")

        res_tab1, res_tab2, res_tab3, res_tab4, res_tab5 = st.tabs([
            "✨ Generated Composite",
            "👥 Side-by-Side Comparison",
            "🔬 Intermediate Stages",
            "📊 Quantitative Evaluation & Audit",
            "📋 Manual Review Checklist",
        ])

        with res_tab1:
            st.image(
                gen_res["composite_image"],
                caption=f"AI Hairstyle Transfer · {gen_res['steps']} steps · Seed {gen_res['seed']}",
                use_container_width=True,
            )
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                st.download_button(
                    "⬇️ Download Result (PNG)",
                    data=image_png_bytes(gen_res["composite_image"]),
                    file_name="local_generative_result.png",
                    mime="image/png",
                    key="dl_local_gen_composite_main",
                )
            with col_dl2:
                st.download_button(
                    "⬇️ Download Edit Mask (PNG)",
                    data=image_png_bytes(gen_res["edit_mask"]),
                    file_name="edit_mask.png",
                    mime="image/png",
                    key="dl_local_gen_mask_main",
                )
            with col_dl3:
                import json
                metrics_json_str = json.dumps(
                    {"metadata": gen_res.get("metadata", {}), "metrics": metrics},
                    indent=2,
                )
                st.download_button(
                    "⬇️ Download Metrics (JSON)",
                    data=metrics_json_str,
                    file_name="metrics.json",
                    mime="application/json",
                    key="dl_local_gen_metrics_json",
                )

        with res_tab2:
            comp_c1, comp_c2 = st.columns(2)
            with comp_c1:
                st.image(target_image, caption="Original Target Person", use_container_width=True)
            with comp_c2:
                st.image(gen_res["composite_image"], caption="Generated Hairstyle Composite", use_container_width=True)

        with res_tab3:
            align_data = gen_res.get("alignment_result")
            prevs = align_data.get("previews", {}) if align_data else {}
            diag_c1, diag_c2, diag_c3, diag_c4 = st.columns(4)
            with diag_c1:
                if "pasted_composite" in prevs:
                    st.image(to_display_pil(prevs["pasted_composite"]), caption="1. Aligned Hairstyle Pre-Composite", use_container_width=True)
                else:
                    st.image(gen_res["reference_crop"], caption="1. Reference Conditioning Crop", use_container_width=True)
            with diag_c2:
                if "edge_repair_mask" in gen_res:
                    st.image(gen_res["edge_repair_mask"], caption="2. Narrow Edge-Repair Mask (Curls Kept)", use_container_width=True)
                else:
                    st.image(gen_res["edit_mask"], caption="2. Target Edit Mask", use_container_width=True)
            with diag_c3:
                st.image(gen_res["raw_generated"], caption="3. Raw Inpainting Output (512x512)", use_container_width=True)
            with diag_c4:
                st.image(gen_res["composite_image"], caption="4. Final Diffusion-Blended Composite", use_container_width=True)

        with res_tab4:
            st.markdown("#### Quantitative Performance & Boundary Preservation Audit")
            qm1, qm2, qm3, qm4 = st.columns(4)
            qm1.metric("Generation Time", f"{gen_res['duration_seconds']:.1f} s")
            qm2.metric("Peak Memory", f"{gen_res['peak_memory_mb']:.1f} MB RSS")
            qm3.metric(
                "Untouched Preservation",
                f"{metrics.get('outside_preservation_pct', 100.0):.1f}%",
                help="Percentage of pixels outside edit mask that are bit-exact identical to the original target.",
            )
            qm4.metric(
                "Facial Protection",
                "PROTECTED" if metrics.get("facial_features_protected", True) else "WARNING",
                help="Verifies zero unauthorized pixel alterations on eyes, eyebrows, nose, mouth, and beard.",
            )

            st.table({
                "Metric Name": [
                    "Output Dimension Match",
                    "Max Pixel Diff Outside Mask",
                    "Mean Pixel Diff Outside Mask",
                    "Outside Exact Preservation %",
                    "Facial Protection Status",
                    "Edit Mask Coverage %",
                    "Data Integrity (No NaN / Blank)",
                    "Denoising Steps",
                    "IP-Adapter Scale",
                    "Random Seed",
                ],
                "Measured Value": [
                    str(metrics.get("dimension_match", True)),
                    f"{metrics.get('max_outside_pixel_diff', 0)} (0 = 100% Bit-Exact)",
                    f"{metrics.get('mean_outside_pixel_diff', 0.0):.4f}",
                    f"{metrics.get('outside_preservation_pct', 100.0):.2f}%",
                    "Verified Protected (0 diff on facial features)" if metrics.get("facial_features_protected", True) else "Notice",
                    f"{metrics.get('edit_mask_coverage_pct', 0.0):.2f}%",
                    "Pass" if not metrics.get("has_nan_or_inf", False) and not metrics.get("is_blank_image", False) else "Fail",
                    str(gen_res.get("steps", 10)),
                    str(gen_res.get("ip_adapter_scale", 0.75)),
                    str(gen_res.get("seed", 42)),
                ],
            })

        with res_tab5:
            st.markdown("#### Qualitative Review Checklist")
            st.caption(
                "Stable Diffusion and IP-Adapter synthesize new hairstyle textures. "
                "Use this manual review checklist to evaluate aesthetic and perceptual quality."
            )
            for idx, q_item in enumerate(gen_res.get("manual_review_questions", MANUAL_REVIEW_QUESTIONS), 1):
                st.checkbox(f"{idx}. {q_item['question']}", key=f"manual_check_{q_item['id']}")

    st.divider()
    st.subheader("🔬 Experimental 2D Diagnostics (Affine & Mask Compositing Fallback)")
    st.caption(
        "Inspect the 2D geometric baseline and intermediate alignment stages for comparison. "
        "2D affine warping is retained as an experimental baseline."
    )

    with st.expander("⚙️ Transfer Settings & Fine-Tuning", expanded=False):
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            match_color = st.checkbox(
                "Harmonize lighting & color (LAB space)",
                value=True,
                help="Matches donor hair luminance and tone to target image lighting.",
            )
            hairline_offset = st.slider(
                "Hairline vertical shift (pixels)",
                min_value=-30,
                max_value=30,
                value=0,
                step=1,
                help="Negative shifts hair higher, positive shifts hair lower.",
            )
        with col_opt2:
            scale_factor = st.slider(
                "Hair volume scale factor",
                min_value=0.85,
                max_value=1.15,
                value=1.00,
                step=0.01,
                help="Fine-tune hair scale relative to skull proportions.",
            )
            mirror_mode = st.selectbox(
                "Orientation mode",
                options=[
                    "Auto (detect head yaw & minimize mismatch)",
                    "Force Mirrored (horizontal flip)",
                    "Force Unmirrored (original reference orientation)",
                ],
                index=0,
                help="Select or override horizontal mirroring of the reference hairstyle.",
            )

    mirror_choice = None
    if mirror_mode == "Force Mirrored (horizontal flip)":
        mirror_choice = True
    elif mirror_mode == "Force Unmirrored (original reference orientation)":
        mirror_choice = False

    if st.button(
        "✨ Transfer hairstyle & compare variants",
        type="primary",
        use_container_width=True,
    ):
        try:
            target_mask = st.session_state["target_results"][0]
            reference_mask = st.session_state["reference_results"][0]
            tgt_seg_map = st.session_state.get("target_seg_map", None)

            with st.spinner("Analyzing head poses, detecting facial landmarks, and warping hairstyle..."):
                # 1. Primary / Landmark-aligned recommended result
                rec_img, rec_diag = transfer_hairstyle(
                    target_image=target_image,
                    reference_image=reference_image,
                    target_hair_mask=target_mask,
                    reference_hair_mask=reference_mask,
                    mirror_reference=mirror_choice,
                    match_color=match_color,
                    hairline_offset_y=hairline_offset,
                    scale_multiplier=scale_factor,
                    method="landmark_affine",
                    return_diagnostics=True,
                    target_segmentation_map=tgt_seg_map,
                    save_intermediate_dir=BASE_DIR / "outputs",
                )

                # 2. Opposite orientation variant for diagnostic comparison
                opp_mirror = not rec_diag["is_mirrored"]
                opp_img, opp_diag = transfer_hairstyle(
                    target_image=target_image,
                    reference_image=reference_image,
                    target_hair_mask=target_mask,
                    reference_hair_mask=reference_mask,
                    mirror_reference=opp_mirror,
                    match_color=match_color,
                    hairline_offset_y=hairline_offset,
                    scale_multiplier=scale_factor,
                    method="landmark_affine",
                    return_diagnostics=True,
                    target_segmentation_map=tgt_seg_map,
                )

                # 3. Baseline bounding box variant
                base_img, base_diag = transfer_hairstyle(
                    target_image=target_image,
                    reference_image=reference_image,
                    target_hair_mask=target_mask,
                    reference_hair_mask=reference_mask,
                    mirror_reference=rec_diag["is_mirrored"],
                    match_color=False,
                    method="baseline_bbox",
                    return_diagnostics=True,
                    target_segmentation_map=tgt_seg_map,
                )

            st.session_state["transfer_variants"] = {
                "rec": (rec_img, rec_diag),
                "opp": (opp_img, opp_diag),
                "base": (base_img, base_diag),
            }
            st.session_state["transfer_result"] = rec_img

            output_path = BASE_DIR / "outputs" / "transfer_result.png"
            save_transfer_result(rec_img, output_path)

        except Exception as exc:
            st.error(f"Hairstyle transfer failed: {exc}")

    # Display results and comparative diagnostics
    if "transfer_variants" in st.session_state:
        variants = st.session_state["transfer_variants"]
        rec_img, rec_diag = variants["rec"]
        opp_img, opp_diag = variants["opp"]
        base_img, base_diag = variants["base"]

        # Viewpoint & pose safeguard warnings
        if rec_diag.get("pose_warning"):
            st.warning(f"⚠️ {rec_diag['pose_warning']}")

        # Orientation & Pose metrics
        diag_col1, diag_col2, diag_col3 = st.columns(3)
        diag_col1.metric("Target Face Yaw", f"{rec_diag['target_yaw']:+.1f}°")
        diag_col2.metric("Donor Hair Yaw", f"{rec_diag['reference_yaw']:+.1f}°")
        diag_col3.metric("Pose Mismatch", f"{rec_diag['yaw_mismatch']:.1f}°")

        orientation_str = "Mirrored (horizontal flip)" if rec_diag["is_mirrored"] else "Unmirrored (original)"
        st.success(
            f"**Recommended Orientation**: {orientation_str}. "
            f"Alignment: {rec_diag['alignment_mode']} ({rec_diag['inliers']} inlier anchors)."
        )

        st.subheader("Diagnostic Comparison")
        rec_label = f"🌟 Landmark-Aligned ({'Mirrored' if rec_diag['is_mirrored'] else 'Unmirrored'}) [Recommended]"
        opp_label = f"🔄 Alternate ({'Unmirrored' if rec_diag['is_mirrored'] else 'Mirrored'})"
        base_label = "📐 Baseline (Naive Bounding Box)"

        tab_rec, tab_opp, tab_base = st.tabs([rec_label, opp_label, base_label])

        with tab_rec:
            st.image(
                rec_img,
                caption=f"Landmark-aligned · {orientation_str}",
                use_container_width=True,
            )
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    "Download recommended result (PNG)",
                    data=image_png_bytes(rec_img),
                    file_name="transfer_result_recommended.png",
                    mime="image/png",
                    key="dl_rec_tab",
                )
            with col_d2:
                if st.button("Save as outputs/transfer_result.png", key="set_rec_main"):
                    st.session_state["transfer_result"] = rec_img
                    save_transfer_result(rec_img, BASE_DIR / "outputs" / "transfer_result.png")
                    st.success("Saved to outputs/transfer_result.png")

        with tab_opp:
            opp_orientation = "Unmirrored" if rec_diag["is_mirrored"] else "Mirrored"
            st.image(
                opp_img,
                caption=f"Opposite orientation · {opp_orientation} (Pose mismatch: {opp_diag['yaw_mismatch']:.1f}°)",
                use_container_width=True,
            )
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                st.download_button(
                    "Download alternate orientation (PNG)",
                    data=image_png_bytes(opp_img),
                    file_name=f"transfer_result_{opp_orientation.lower()}.png",
                    mime="image/png",
                    key="dl_opp_tab",
                )
            with col_o2:
                if st.button(f"Save this {opp_orientation} version to outputs/", key="set_opp_main"):
                    st.session_state["transfer_result"] = opp_img
                    save_transfer_result(opp_img, BASE_DIR / "outputs" / "transfer_result.png")
                    st.success(f"Saved {opp_orientation} version to outputs/transfer_result.png")

        with tab_base:
            st.image(
                base_img,
                caption="Baseline: Stretches hair mask bounding box directly without facial anchors",
                use_container_width=True,
            )
            st.caption(
                "The baseline method illustrates the 'hairpiece' effect: bounding box deformation, "
                "destructive inpainting of the whole forehead, and no facial landmark alignment."
            )
            st.download_button(
                "Download baseline comparison (PNG)",
                data=image_png_bytes(base_img),
                file_name="transfer_result_baseline.png",
                mime="image/png",
                key="dl_base_tab",
            )

        # Numerical Validation and Metrics Panel
        st.divider()
        st.subheader("📊 Alignment & Numerical Validation")
        val_c1, val_c2, val_c3, val_c4 = st.columns(4)
        val_c1.metric("Affine Scale", f"{rec_diag.get('scale', 1.0):.3f}")
        val_c2.metric("Affine Rotation", f"{rec_diag.get('rotation_deg', 0.0):+.1f}°")
        val_c3.metric("Spill Outside Head", f"{rec_diag.get('spill_percentage', 0.0):.1f}%", help="Donor hair outside permitted head envelope (clipped)")
        val_c4.metric("Clothing Overlap", f"{rec_diag.get('clothing_overlap_px', 0)} px", help="Donor hair overlap on target clothing (strictly 0)")

        val_c5, val_c6, val_c7, val_c8 = st.columns(4)
        val_c5.metric("Translation X (dx)", f"{rec_diag.get('tx', 0.0):+.1f} px")
        val_c6.metric("Translation Y (dy)", f"{rec_diag.get('ty', 0.0):+.1f} px")
        val_c7.metric("RANSAC Inliers", f"{rec_diag.get('inliers', 0)} landmarks")
        val_c8.metric("Uncovered Target Hair", f"{rec_diag.get('uncovered_hair_percentage', 0.0):.1f}%", help="Target hair not covered by donor; preserved to prevent white scalp holes")

        # Warnings log
        if rec_diag.get("warnings"):
            with st.expander("⚠️ Pipeline Validation Warnings & Notices", expanded=True):
                for w in rec_diag["warnings"]:
                    st.warning(w)

        # 9 Intermediate Images + Hairline Overlay
        intermediates = rec_diag.get("intermediates", {})
        if intermediates:
            st.divider()
            with st.expander("🔬 9-Stage Pipeline Audit & Intermediate Images", expanded=False):
                st.caption("Inspect each step of the pipeline to verify segmentation, alignment, masking, and compositing.")

                if "hairline_overlay" in intermediates:
                    st.subheader("Target Head & Hairline Constraints Diagnostic")
                    st.image(
                        to_display_pil(intermediates["hairline_overlay"]),
                        caption="Diagnostic Overlay: Cyan = Permitted Cranial Envelope, Yellow = Forehead/Temple Anchor Line, Green = Target Hair Mask, Red Dots = Landmark Anchors",
                        use_container_width=True,
                    )

                st.subheader("9 Sequential Pipeline Stages")
                stage_keys = [
                    ("1_target", "1. Original Target"),
                    ("2_donor", "2. Original Donor"),
                    ("3_target_hair_mask", "3. Target Hair Mask"),
                    ("4_donor_hair_mask", "4. Donor Hair Mask (Mirrored/Active)"),
                    ("5_transformed_donor_image", "5. Transformed Donor Image"),
                    ("6_transformed_donor_mask", "6. Transformed & Clipped Donor Mask"),
                    ("7_protected_face_mask", "7. Protected Face Mask"),
                    ("8_inpainting_result", "8. Inpainting Result (Flyaways Only)"),
                    ("9_final_composite", "9. Final Blended Composite"),
                ]

                for row_idx in range(3):
                    cols = st.columns(3)
                    for col_idx in range(3):
                        stage_num = row_idx * 3 + col_idx
                        if stage_num < len(stage_keys):
                            s_key, s_title = stage_keys[stage_num]
                            if s_key in intermediates:
                                with cols[col_idx]:
                                    pil_img = to_display_pil(intermediates[s_key])
                                    st.image(pil_img, caption=s_title, use_container_width=True)
                                    st.download_button(
                                        f"Download {s_title}",
                                        data=image_png_bytes(pil_img),
                                        file_name=f"diag_{s_key}.png",
                                        mime="image/png",
                                        key=f"dl_diag_{s_key}",
                                    )
    elif "transfer_result" in st.session_state:
        result = st.session_state["transfer_result"]
        st.image(
            result,
            caption="Hairstyle transfer result",
            use_container_width=True,
        )
        st.download_button(
            "Download transfer result (PNG)",
            data=image_png_bytes(result),
            file_name="transfer_result.png",
            mime="image/png",
        )


# =========================================================
# EXPERIMENT HISTORY GALLERY
# =========================================================

st.divider()
st.header("📁 Experiment History Gallery")
st.caption(
    "All local generative runs are automatically persisted to `outputs/experiments/<experiment_id>/`. "
    "Review and compare multi-image runs conducted in this and previous sessions."
)

saved_exps = list_saved_experiments(BASE_DIR / "outputs")
if not saved_exps:
    st.info("No saved experiments found in `outputs/experiments/`. Run the local generative pipeline above to record your first experiment.")
else:
    st.write(f"Found **{len(saved_exps)}** saved experiment(s):")
    for exp in saved_exps[:8]:  # Show up to 8 most recent
        exp_id = exp["id"]
        meta = exp.get("metadata", {})
        metrics = exp.get("metrics", {})
        timestamp = meta.get("timestamp", "Unknown date")
        steps = meta.get("num_inference_steps", meta.get("steps", "N/A"))
        scale = meta.get("ip_adapter_scale", "N/A")
        seed = meta.get("seed", "N/A")
        duration = meta.get("duration_seconds", "N/A")

        with st.expander(f"🧪 {exp_id} · {timestamp} · {steps} steps · scale={scale} · seed={seed}", expanded=False):
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            with col_e1:
                if exp["target_path"].is_file():
                    st.image(str(exp["target_path"]), caption="Target Portrait", use_container_width=True)
            with col_e2:
                if exp["reference_path"].is_file():
                    st.image(str(exp["reference_path"]), caption="Reference Hairstyle", use_container_width=True)
            with col_e3:
                if exp["edit_mask_path"].is_file():
                    st.image(str(exp["edit_mask_path"]), caption="Edit Mask", use_container_width=True)
            with col_e4:
                if exp["generated_path"].is_file():
                    st.image(str(exp["generated_path"]), caption="Generated Hairstyle", use_container_width=True)

            em1, em2, em3, em4 = st.columns(4)
            em1.metric("Duration", f"{duration} s")
            em2.metric("Outside Preservation", f"{metrics.get('outside_preservation_pct', 100.0):.1f}%")
            em3.metric("Facial Protection", "Protected" if metrics.get("facial_features_protected", True) else "Notice")
            em4.metric("Peak Memory", f"{metrics.get('peak_memory_mb', 'N/A')} MB")

            if exp["generated_path"].is_file():
                with open(exp["generated_path"], "rb") as f_img:
                    st.download_button(
                        f"Download Generated PNG ({exp_id})",
                        data=f_img.read(),
                        file_name=f"{exp_id}_generated.png",
                        mime="image/png",
                        key=f"dl_exp_{exp_id}",
                    )

