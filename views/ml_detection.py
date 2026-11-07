"""
views/ml_detection.py
=====================
Experimental Bitemporal Deep-Learning Change Detection (U-Net).
Processes 6-channel bi-temporal imagery through trained Space Netra checkpoints.
Features dynamic threshold control, raw prediction visualization without dilation,
and dataset-level benchmark transparency.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

import spacenetra_model
import photo_streamer
import city_catalog
from views.theme import render_page_header

@st.cache_resource
def get_cached_model(path: str):
    """Loads and strictly validates model checkpoint."""
    return spacenetra_model.load_model(path)

def render_ml_detection():
    """Renders the ML change detection module."""
    render_page_header(
        title="ML Change Detection — Bitemporal U-Net Architecture",
        subtitle="Experimental 6-Channel Deep-Learning Change Detection with Dynamic Sigmoid Threshold Control",
        badge_text="Deep Learning Pipeline",
        badge_type="ready"
    )

    # Checkpoint configuration
    orig_path = spacenetra_model.INITIAL_CHECKPOINT_PATH
    fulltrain_path = spacenetra_model.FULLTRAIN_CHECKPOINT_PATH
    mixed_path = spacenetra_model.MIXED_CHECKPOINT_PATH

    title_orig = "1. S2Looking baseline — original checkpoint"
    title_fulltrain = "2. S2Looking full-training model"
    title_mixed = "3. S2Looking + LEVIR-CD mixed model"

    available_options = []
    if os.path.exists(orig_path):
        available_options.append(title_orig)
    if os.path.exists(fulltrain_path):
        available_options.append(title_fulltrain)
    if os.path.exists(mixed_path):
        available_options.append(title_mixed)

    title_to_key = {
        title_orig: "initial",
        title_fulltrain: "fulltrain",
        title_mixed: "mixed",
        "S2Looking baseline — original checkpoint": "initial",
        "S2Looking full-training model": "fulltrain",
        "S2Looking + LEVIR-CD mixed model": "mixed"
    }

    default_opt_idx = available_options.index(title_mixed) if title_mixed in available_options else 0

    # ── Sidebar Controls (Isolated to ML page) ──
    st.sidebar.markdown("<div class='workstation-card-title'>Model & Inference Settings</div>", unsafe_allow_html=True)

    ml_eval_mode = st.sidebar.radio(
        "Evaluation Mode:",
        ["Single Checkpoint Evaluation", "Side-by-Side Dual-Model Comparison"],
        help="Evaluate one checkpoint or compare two checkpoints on identical inputs."
    )

    if ml_eval_mode == "Single Checkpoint Evaluation":
        selected_ckpt_title = st.sidebar.selectbox(
            "Model Checkpoint:",
            available_options,
            index=default_opt_idx,
            help="S2Looking + LEVIR-CD mixed model is the default experimental building-change model."
        )
        selected_ckpt_key = title_to_key.get(selected_ckpt_title, "mixed")
        ckpt_cfg = spacenetra_model.CHECKPOINTS_CONFIG[selected_ckpt_key]
        selected_ckpt_path = ckpt_cfg["path"]
        ml_model, ml_meta, ml_err = get_cached_model(selected_ckpt_path)
    else:
        comp_title_a = st.sidebar.selectbox("Model A (Primary):", available_options, index=default_opt_idx, key="ml_sel_a")
        idx_b = available_options.index(title_fulltrain) if title_fulltrain in available_options else 0
        comp_title_b = st.sidebar.selectbox("Model B (Comparative):", available_options, index=idx_b, key="ml_sel_b")

        key_a = title_to_key.get(comp_title_a, "mixed")
        key_b = title_to_key.get(comp_title_b, "fulltrain")
        cfg_a = spacenetra_model.CHECKPOINTS_CONFIG[key_a]
        cfg_b = spacenetra_model.CHECKPOINTS_CONFIG[key_b]
        ckpt_cfg = cfg_a
        selected_ckpt_path = cfg_a["path"]
        ml_model, ml_meta, ml_err = None, None, None

    thresh_val = st.sidebar.slider(
        "Decision Threshold (p ≥ τ):",
        min_value=0.05,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Applies directly to sigmoid probabilities to produce binary mask without inference re-runs."
    )

    ml_source_mode = st.sidebar.radio(
        "Imagery Source:",
        ["Active Municipal Sector", "Upload Custom Pair"],
        help="Choose between cached municipal satellite imagery or custom bitemporal photos."
    )

    # ── Dataset-Level Benchmark Evaluation Cards ──
    st.markdown("##### Dataset-Level Benchmark Evaluation Results")
    st.caption("Standardized test set performance evaluated on held-out benchmark datasets at decision threshold 0.50.")

    bm1, bm2 = st.columns(2)
    with bm1:
        st.markdown("""
        <div class="workstation-card">
            <div style="font-weight: 700; color: #38bdf8; font-size: 0.90rem; margin-bottom: 6px;">
                S2Looking Test Benchmark
            </div>
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
                • <strong>Precision:</strong> 33.04%<br>
                • <strong>Recall:</strong> 30.50%<br>
                • <strong>F1 Score:</strong> 31.72%<br>
                • <strong>IoU:</strong> 18.85%<br>
                <span style="color: #64748b; font-size: 0.76rem;">Threshold used: 0.50</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with bm2:
        st.markdown("""
        <div class="workstation-card">
            <div style="font-weight: 700; color: #10b981; font-size: 0.90rem; margin-bottom: 6px;">
                LEVIR-CD Test Benchmark
            </div>
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
                • <strong>Precision:</strong> 68.16%<br>
                • <strong>Recall:</strong> 63.49%<br>
                • <strong>F1 Score:</strong> 65.75%<br>
                • <strong>IoU:</strong> 48.97%<br>
                <span style="color: #64748b; font-size: 0.76rem;">Threshold used: 0.50</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 6px; padding: 8px 14px; margin-bottom: 16px; font-size: 0.80rem; color: #cbd5e1;">
        <strong>Evaluation Scope Notice:</strong> These are separate dataset-level benchmark results, not Maharashtra-specific accuracy.
    </div>
    """, unsafe_allow_html=True)

    # ── Imagery Input & Provenance Resolution ──
    input_img1 = None
    input_img2 = None
    is_custom = (ml_source_mode == "Upload Custom Pair")

    if not is_custom:
        area_name = st.session_state.get("active_area_name", "CIDCO Urban Expansion (N-1 to N-12)")
        h_info = city_catalog.get_hotspot_info(area_name)
        tl = photo_streamer.fetch_3stage_photographic_timeline(h_info["lat"], h_info["lon"], zoom=h_info.get("zoom", 17))
        input_img1 = tl["p1_path"]
        input_img2 = tl["p3_path"]
        sig = f"sec_{area_name}_{tl.get('dates',{}).get('epoch_1')}_{tl.get('dates',{}).get('epoch_3')}"

        if tl.get("is_real"):
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 6px; padding: 8px 14px; margin-bottom: 14px;">
                <span style="color: #10b981; font-weight: 700; font-size: 0.82rem;">GENUINE SATELLITE IMAGERY</span>
                <span style="color: #94a3b8; font-size: 0.78rem; margin-left: 8px;">Verified optical satellite photography (EPSG:4326 georeferenced).</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 6px; padding: 8px 14px; margin-bottom: 14px;">
                <span style="color: #f59e0b; font-weight: 700; font-size: 0.82rem;">SYNTHETIC TEST IMAGERY</span>
                <span style="color: #94a3b8; font-size: 0.78rem; margin-left: 8px;">Procedural simulation fallback fixture (not real observation).</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(148, 163, 184, 0.08); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 6px; padding: 10px 14px; margin-bottom: 14px;">
            <div style="color: #cbd5e1; font-weight: 700; font-size: 0.84rem; margin-bottom: 4px;">
                UNKNOWN-SOURCE / USER-UPLOADED IMAGERY (UNREFERENCED)
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.4;">
                Uploaded imagery lacks embedded geospatial coordinate reference systems (CRS) and ground sampling distance (GSD).
                Coordinates and physical ground area (m² or hectares) are not computed.
                Predicted differences represent pixel-level binary change probability, not confirmed construction or demolition.
            </div>
        </div>
        """, unsafe_allow_html=True)

        uc1, uc2 = st.columns(2)
        with uc1:
            f1 = st.file_uploader("Earlier Image (T1):", type=["png", "jpg", "jpeg"], key="ml_upl_1")
        with uc2:
            f2 = st.file_uploader("Later Image (T2):", type=["png", "jpg", "jpeg"], key="ml_upl_2")

        if f1 and f2:
            input_img1 = Image.open(f1)
            input_img2 = Image.open(f2)
            sig = f"upl_{getattr(f1, 'name', '')}_{getattr(f2, 'name', '')}"
        else:
            sig = "upl_none"
            st.info("Upload both images to activate model inference.")

    run_ml = st.button("Run ML Change Detection", type="primary", disabled=(input_img1 is None or input_img2 is None))

    # ── BRANCH 1: Single Checkpoint Evaluation ──
    if ml_eval_mode == "Single Checkpoint Evaluation":
        if run_ml and input_img1 is not None and input_img2 is not None and ml_model is not None:
            with st.spinner(f"Running inference with {ckpt_cfg['short_label']}..."):
                t_6ch, im1_prep, im2_prep = spacenetra_model.preprocess_pair(input_img1, input_img2, target_size=256)
                t_in = t_6ch.to(ml_meta["device"])
                with torch.no_grad():
                    logits = ml_model(t_in)
                    probs = torch.sigmoid(logits)
                prob_arr = probs.squeeze().cpu().numpy().astype(np.float32)

                st.session_state["ml_prob_map"] = prob_arr
                st.session_state["ml_im1_prep"] = im1_prep
                st.session_state["ml_im2_prep"] = im2_prep
                st.session_state["ml_checkpoint_path"] = selected_ckpt_path
                st.session_state["ml_sig"] = sig

        has_pred = (
            "ml_prob_map" in st.session_state and
            st.session_state.get("ml_checkpoint_path") == selected_ckpt_path and
            st.session_state.get("ml_sig") == sig
        )

        if has_pred:
            prob_map = st.session_state["ml_prob_map"]
            im1_p = st.session_state["ml_im1_prep"]
            im2_p = st.session_state["ml_im2_prep"]

            pred = spacenetra_model.apply_threshold_to_prob_map(prob_map, im2_p, threshold=thresh_val)

            st.markdown("---")
            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Predicted Changed Pixels", f"{pred['changed_pixels']:,} px")
            pm2.metric("Changed Frame Coverage", f"{pred['change_ratio_pct']}% of frame")
            pm3.metric("Decision Threshold", f"{thresh_val:.2f}")
            pm4.metric("Active Model", ckpt_cfg["short_label"])

            # Visual Pipeline
            st.markdown("##### Visual Prediction Pipeline")
            v1, v2, v3, v4, v5 = st.columns(5)
            with v1:
                st.image(im1_p, caption="Earlier Image (T1, 256×256)", width="stretch")
            with v2:
                st.image(im2_p, caption="Later Image (T2, 256×256)", width="stretch")
            with v3:
                st.image(pred["prob_heatmap"], caption="Probability Map (Sigmoid)", width="stretch")
            with v4:
                st.image(pred["binary_mask"] * 255, caption=f"Binary Mask (τ ≥ {thresh_val:.2f})", width="stretch")
            with v5:
                st.image(pred["overlay_img"], caption="Change Overlay", width="stretch")

            st.caption("Coral Red = Detected Change (Probability ≥ Threshold) | Turbo Gradient = Continuous Probability (0.0 to 1.0) | Natural Image = Unchanged Ground. Raw predictions are preserved without automatic morphological expansion or dilation.")

            # Sensitivity Table
            st.markdown("---")
            st.markdown("##### Threshold Sensitivity Diagnostic")
            st.caption("Calculates predicted changed pixels and coverage across standard decision thresholds without re-running inference.")

            sens_diag = spacenetra_model.compute_threshold_sensitivity_diagnostic(
                prob_map, thresholds=[0.10, 0.20, 0.30, 0.40, 0.50, 0.65]
            )

            diag_rows = []
            for s in sens_diag:
                th = s["threshold"]
                regime = (
                    "High Recall / Exploratory" if th <= 0.20
                    else "Standard Decision Boundary" if th <= 0.50
                    else "High Precision / Conservative"
                )
                diag_rows.append({
                    "Threshold": f"{th:.2f}",
                    "Predicted Changed Pixels": f"{s['changed_pixels']:,} px",
                    "Frame Percentage": f"{s['change_pct']:.2f}%",
                    "Operating Regime": regime
                })
            st.dataframe(pd.DataFrame(diag_rows), width="stretch", hide_index=True)

            g_cols = st.columns(6)
            for idx, s in enumerate(sens_diag):
                with g_cols[idx]:
                    st.image(
                        s["binary_mask"] * 255,
                        caption=f"τ ≥ {s['threshold']:.2f}\n{s['changed_pixels']:,} px ({s['change_pct']}%)",
                        width="stretch"
                    )

    # ── BRANCH 2: Dual-Model Comparison ──
    else:
        model_a, meta_a, _ = get_cached_model(cfg_a["path"])
        model_b, meta_b, _ = get_cached_model(cfg_b["path"])

        if run_ml and input_img1 is not None and input_img2 is not None:
            with st.spinner(f"Comparing {cfg_a['short_label']} and {cfg_b['short_label']}..."):
                t_6ch, im1_prep, im2_prep = spacenetra_model.preprocess_pair(input_img1, input_img2, target_size=256)
                t_in_a = t_6ch.to(meta_a["device"])
                t_in_b = t_6ch.to(meta_b["device"])
                with torch.no_grad():
                    logits_a = model_a(t_in_a)
                    probs_a = torch.sigmoid(logits_a)
                    logits_b = model_b(t_in_b)
                    probs_b = torch.sigmoid(logits_b)

                st.session_state["ml_dual_a"] = probs_a.squeeze().cpu().numpy().astype(np.float32)
                st.session_state["ml_dual_b"] = probs_b.squeeze().cpu().numpy().astype(np.float32)
                st.session_state["ml_dual_im1"] = im1_prep
                st.session_state["ml_dual_im2"] = im2_prep
                st.session_state["ml_dual_sig"] = sig
                st.session_state["ml_dual_ka"] = key_a
                st.session_state["ml_dual_kb"] = key_b

        has_dual = (
            "ml_dual_a" in st.session_state and
            "ml_dual_b" in st.session_state and
            st.session_state.get("ml_dual_sig") == sig and
            st.session_state.get("ml_dual_ka") == key_a and
            st.session_state.get("ml_dual_kb") == key_b
        )

        if has_dual:
            prob_a = st.session_state["ml_dual_a"]
            prob_b = st.session_state["ml_dual_b"]
            im1_p = st.session_state["ml_dual_im1"]
            im2_p = st.session_state["ml_dual_im2"]

            pred_a = spacenetra_model.apply_threshold_to_prob_map(prob_a, im2_p, threshold=thresh_val)
            pred_b = spacenetra_model.apply_threshold_to_prob_map(prob_b, im2_p, threshold=thresh_val)

            st.markdown("---")
            st.markdown(f"##### Comparative Inference Results (Threshold {thresh_val:.2f})")

            dm1, dm2, dm3, dm4 = st.columns(4)
            dm1.metric(f"Model A: {cfg_a['short_label']}", f"{pred_a['changed_pixels']:,} px", delta=f"{pred_a['change_ratio_pct']}% frame")
            dm2.metric(f"Model B: {cfg_b['short_label']}", f"{pred_b['changed_pixels']:,} px", delta=f"{pred_b['change_ratio_pct']}% frame")
            rat = f"{pred_a['changed_pixels'] / max(1, pred_b['changed_pixels']):.1f}×" if pred_b['changed_pixels'] > 0 else "N/A"
            dm3.metric("Ratio (A / B)", rat)
            dm4.metric("Applied Threshold", f"{thresh_val:.2f}")

            st.write(f"**Model A ({cfg_a['title']}):**")
            va1, va2, va3, va4 = st.columns(4)
            va1.image(im2_p, caption="Later Image (T2)", width="stretch")
            va2.image(pred_a["prob_heatmap"], caption="Probability Map", width="stretch")
            va3.image(pred_a["binary_mask"] * 255, caption=f"Binary Mask (τ ≥ {thresh_val:.2f})", width="stretch")
            va4.image(pred_a["overlay_img"], caption=f"Overlay ({pred_a['changed_pixels']:,} px)", width="stretch")

            st.write(f"**Model B ({cfg_b['title']}):**")
            vb1, vb2, vb3, vb4 = st.columns(4)
            vb1.image(im2_p, caption="Later Image (T2)", width="stretch")
            vb2.image(pred_b["prob_heatmap"], caption="Probability Map", width="stretch")
            vb3.image(pred_b["binary_mask"] * 255, caption=f"Binary Mask (τ ≥ {thresh_val:.2f})", width="stretch")
            vb4.image(pred_b["overlay_img"], caption=f"Overlay ({pred_b['changed_pixels']:,} px)", width="stretch")

    # ── Advanced Diagnostic Expanders ──
    st.markdown("---")
    if input_img1 is not None and input_img2 is not None:
        with st.expander("Sub-Pixel Alignment & Geometric Registration Diagnostic", expanded=False):
            try:
                diag = spacenetra_model.verify_input_consistency_and_alignment(input_img1, input_img2)
                ac1, ac2, ac3, ac4 = st.columns(4)
                ac1.metric("Dimensions", f"{diag['orig_size_earlier'][0]}×{diag['orig_size_earlier'][1]} px",
                           delta="Compatible" if diag["compatible_framing"] else "Mismatched")
                ac2.metric("Sub-Pixel Offset (dx, dy)", f"{diag['dx_pixels']:+.2f}, {diag['dy_pixels']:+.2f} px")
                ac3.metric("Displacement Magnitude", f"{diag['displacement_magnitude_px']:.2f} px")
                ac4.metric("Spatial Correlation (r)", f"{diag['pearson_correlation_r']:.4f}")
                st.caption(f"Alignment Assessment: {diag['alignment_description']} (Phase Peak: {diag['phase_peak_response']:.4f})")
            except Exception as e:
                st.caption(f"Registration diagnostic unavailable: {e}")
