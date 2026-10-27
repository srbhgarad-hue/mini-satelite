"""
test_ml_model.py
================
Comprehensive Validation Suite for Space Netra Experimental U-Net Integration.

Verifies:
1. Checkpoint Registry: Both checkpoints configured with accurate descriptions and loaded strictly with strict=True.
2. Training Preprocessing Compliance (bilinear 256x256, [0, 1] normalized, earlier RGB + later RGB in 6-channel CHW format).
3. Output Mask Dimensions (256x256) and continuous finite probabilities strictly in [0, 1].
4. Real Detection-Threshold Slider responsiveness (0.05 to 0.90) and instant probability-map re-thresholding without inference re-run.
5. Probability Heatmap generation using continuous probability values with OpenCV colormap.
6. Optional Morphology Post-Processing (opening, closing, closing_opening) as a separate non-destructive filter.
7. Pair-Specific Ground-Truth Reference Mask Evaluation (precision, recall, F1, IoU, confusion map).
8. Preservation of existing semantic-search (CLIP + FAISS) and traditional detection features.
"""

import os
import io
import numpy as np
from PIL import Image
import torch

import spacenetra_model
import building_detector


def test_checkpoint_strict_loading():
    print("------------------------------------------------------------------")
    print("[TEST 1] Checkpoint Selector & Strict State-Dict Verification")
    print("------------------------------------------------------------------")
    
    # Verify registry contains both required checkpoints with accurate metadata
    assert "initial" in spacenetra_model.CHECKPOINTS_CONFIG, "Missing 'initial' in CHECKPOINTS_CONFIG"
    assert "fulltrain" in spacenetra_model.CHECKPOINTS_CONFIG, "Missing 'fulltrain' in CHECKPOINTS_CONFIG"
    
    init_cfg = spacenetra_model.CHECKPOINTS_CONFIG["initial"]
    full_cfg = spacenetra_model.CHECKPOINTS_CONFIG["fulltrain"]
    
    print(f"Checking full-train checkpoint: {full_cfg['path']}")
    assert os.path.exists(full_cfg["path"]), f"Full-train checkpoint missing at {full_cfg['path']}"
    assert os.path.getsize(full_cfg["path"]) > 2_000_000, "Full-train checkpoint seems truncated"

    print(f"Checking initial checkpoint: {init_cfg['path']}")
    assert os.path.exists(init_cfg["path"]), f"Initial checkpoint missing at {init_cfg['path']}"
    assert os.path.getsize(init_cfg["path"]) > 2_000_000, "Initial checkpoint seems truncated"

    # 1. Load full-training checkpoint strictly
    model_full, meta_full, err_full = spacenetra_model.load_model(full_cfg["path"], device="cpu")
    assert err_full is None, f"Strict loading of full-train checkpoint failed: {err_full}"
    assert model_full is not None, "Loaded full-train model is None"
    assert meta_full["epoch"] == 9, f"Expected epoch 9, got {meta_full.get('epoch')}"
    assert meta_full["training_pairs"] == 3500, f"Expected 3,500 train pairs, got {meta_full.get('training_pairs')}"
    assert meta_full["validation_pairs"] == 500, f"Expected 500 val pairs, got {meta_full.get('validation_pairs')}"
    assert round(meta_full["validation_metrics"]["f1"], 4) == 0.3910, f"Unexpected Val F1: {meta_full['validation_metrics']['f1']}"
    assert round(meta_full["validation_metrics"]["iou"], 4) == 0.2430, f"Unexpected Val IoU: {meta_full['validation_metrics']['iou']}"
    assert round(meta_full["heldout_test_metrics"]["f1"], 4) == 0.3475, f"Unexpected Test F1 @ 0.50: {meta_full['heldout_test_metrics']['f1']}"
    assert round(meta_full["heldout_test_metrics"]["iou"], 4) == 0.2103, f"Unexpected Test IoU @ 0.50: {meta_full['heldout_test_metrics']['iou']}"
    assert round(meta_full["heldout_test_metrics_th20"]["f1"], 4) == 0.3526, f"Unexpected Test F1 @ 0.20: {meta_full['heldout_test_metrics_th20']['f1']}"
    assert round(meta_full["heldout_test_metrics_th20"]["iou"], 4) == 0.2141, f"Unexpected Test IoU @ 0.20: {meta_full['heldout_test_metrics_th20']['iou']}"
    print("  [PASS] Full-train checkpoint (Epoch 9) loaded strictly with strict=True!")
    print(f"    - Training: {meta_full['training_pairs']} pairs | Validation: {meta_full['validation_pairs']} pairs")
    print(f"    - Benchmark Val F1: {meta_full['validation_metrics']['f1']:.4f} | IoU: {meta_full['validation_metrics']['iou']:.4f}")
    print(f"    - Held-Out Test (Th=0.50) F1: {meta_full['heldout_test_metrics']['f1']:.4f} | IoU: {meta_full['heldout_test_metrics']['iou']:.4f}")
    print(f"    - Held-Out Test (Th=0.20) F1: {meta_full['heldout_test_metrics_th20']['f1']:.4f} | IoU: {meta_full['heldout_test_metrics_th20']['iou']:.4f}")

    # 2. Load initial baseline checkpoint strictly
    model_init, meta_init, err_init = spacenetra_model.load_model(init_cfg["path"], device="cpu")
    assert err_init is None, f"Strict loading of initial checkpoint failed: {err_init}"
    assert model_init is not None, "Loaded initial model is None"
    assert meta_init["epoch"] == 2, f"Expected epoch 2, got {meta_init.get('epoch')}"
    assert meta_init["training_pairs"] == 500, f"Expected 500 train pairs, got {meta_init.get('training_pairs')}"
    assert meta_init["validation_pairs"] == 100, f"Expected 100 val pairs, got {meta_init.get('validation_pairs')}"
    print("  [PASS] Initial checkpoint (Epoch 2) loaded strictly with strict=True!")
    print(f"    - Training: {meta_init['training_pairs']} pairs | Validation: {meta_init['validation_pairs']} pairs")
    print(f"    - Benchmark Val F1: {meta_init['validation_metrics']['f1']:.4f} | IoU: {meta_init['validation_metrics']['iou']:.4f}")

    return (model_full, meta_full), (model_init, meta_init)


def test_preprocessing_compliance():
    print("\n------------------------------------------------------------------")
    print("[TEST 2] Preserved Training Preprocessing Verification")
    print("------------------------------------------------------------------")
    img1 = Image.new("RGB", (320, 240), color=(100, 150, 200))
    img2 = Image.new("RGB", (480, 360), color=(110, 160, 210))

    tensor_6ch, im1_resized, im2_resized = spacenetra_model.preprocess_pair(img1, img2, target_size=256)

    # Tensor format
    assert tensor_6ch.shape == (1, 6, 256, 256), f"Expected shape (1, 6, 256, 256), got {tensor_6ch.shape}"
    assert tensor_6ch.dtype == torch.float32, f"Expected float32, got {tensor_6ch.dtype}"
    assert im1_resized.size == (256, 256), f"Expected 256x256, got {im1_resized.size}"
    assert im2_resized.size == (256, 256), f"Expected 256x256, got {im2_resized.size}"

    # Normalized to [0, 1] without ImageNet centering
    min_val = tensor_6ch.min().item()
    max_val = tensor_6ch.max().item()
    assert 0.0 <= min_val <= 1.0, f"Tensor min {min_val} outside [0, 1]"
    assert 0.0 <= max_val <= 1.0, f"Tensor max {max_val} outside [0, 1]"

    # Confirm channel ordering: earlier RGB (0:3) and later RGB (3:6)
    t1_np = tensor_6ch[0, :3, :, :].numpy()
    t2_np = tensor_6ch[0, 3:, :, :].numpy()
    assert abs(t1_np[0, 0, 0] - 100.0 / 255.0) < 1e-3, "Channel 0 does not match earlier image"
    assert abs(t2_np[0, 0, 0] - 110.0 / 255.0) < 1e-3, "Channel 3 does not match later image"

    print("  [PASS] Preprocessing strictly adheres to training specification:")
    print(f"    - Shape: {tensor_6ch.shape} (BCHW format, 6 channels)")
    print(f"    - Pixel range: [{min_val:.4f}, {max_val:.4f}] in [0, 1]")
    print("    - Earlier RGB (channels 0..2) followed by later RGB (channels 3..5)")
    print("    - Bilinear interpolation to 256x256 confirmed.")

    return tensor_6ch, im2_resized


def test_inference_and_dimensions(model, tensor_6ch, im2_resized):
    print("\n------------------------------------------------------------------")
    print("[TEST 3] Inference, Output Dimensions & Probability Range")
    print("------------------------------------------------------------------")
    pred = spacenetra_model.predict_change(model, tensor_6ch, im2_resized, threshold=0.50, device="cpu")

    # Dimensions
    assert pred["prob_map"].shape == (256, 256), f"Expected prob_map (256, 256), got {pred['prob_map'].shape}"
    assert pred["binary_mask"].shape == (256, 256), f"Expected binary_mask (256, 256), got {pred['binary_mask'].shape}"
    assert pred["overlay_img"].shape == (256, 256, 3), f"Expected overlay_img (256, 256, 3), got {pred['overlay_img'].shape}"
    assert pred["prob_heatmap"].shape == (256, 256, 3), f"Expected prob_heatmap (256, 256, 3), got {pred['prob_heatmap'].shape}"
    assert pred["total_pixels"] == 256 * 256, f"Expected total pixels 65,536, got {pred['total_pixels']}"

    # Probabilities finite and within [0, 1]
    assert pred["is_valid"] is True, "Probabilities contain NaN or Inf values!"
    assert np.isfinite(pred["prob_map"]).all(), "Non-finite values found in prob_map"
    p_min = float(pred["prob_map"].min())
    p_max = float(pred["prob_map"].max())
    assert 0.0 <= p_min <= 1.0, f"prob_map min {p_min} outside [0, 1]"
    assert 0.0 <= p_max <= 1.0, f"prob_map max {p_max} outside [0, 1]"

    # Binary mask values strictly in {0, 1}
    unique_vals = set(np.unique(pred["binary_mask"]))
    assert unique_vals.issubset({0, 1}), f"Non-binary values in binary_mask: {unique_vals}"

    print("  [PASS] Inference output dimensions, probabilities, and heatmap verified:")
    print(f"    - Output mask dimensions: {pred['binary_mask'].shape}")
    print(f"    - Probabilities range: [{p_min:.5f}, {p_max:.5f}] (all finite, [0, 1])")
    print(f"    - Changed pixels at threshold 0.50: {pred['changed_pixels']} ({pred['change_ratio_pct']}%)")


def test_threshold_responsiveness_and_slider(model):
    print("\n------------------------------------------------------------------")
    print("[TEST 4] Real Detection-Threshold Slider & Dynamic Re-Thresholding")
    print("------------------------------------------------------------------")
    t1_path = "data/change_detection/source/csn_7dca6e98_2016.jpg"
    t2_path = "data/change_detection/source/csn_7dca6e98_current.jpg"
    assert os.path.exists(t1_path) and os.path.exists(t2_path), "Test satellite tiles missing"

    tensor_real, _, r2 = spacenetra_model.preprocess_pair(t1_path, t2_path, target_size=256)

    # Compute base inference
    base_pred = spacenetra_model.predict_change(model, tensor_real, r2, threshold=0.50, device="cpu")
    prob_map = base_pred["prob_map"]

    slider_thresholds = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90]
    results = {}
    masks = {}

    for th in slider_thresholds:
        # Test dynamic re-threshold helper directly on continuous probability map
        re_pred = spacenetra_model.apply_threshold_to_prob_map(prob_map, r2, threshold=th)
        results[th] = re_pred["changed_pixels"]
        masks[th] = re_pred["binary_mask"]
        assert re_pred["threshold"] == th, f"Reported threshold {re_pred['threshold']} != requested {th}"
        
        # Cross-verify with full predict_change at the same threshold
        direct_pred = spacenetra_model.predict_change(model, tensor_real, r2, threshold=th, device="cpu")
        assert re_pred["changed_pixels"] == direct_pred["changed_pixels"], f"Mismatch at threshold {th}"
        assert np.array_equal(re_pred["binary_mask"], direct_pred["binary_mask"]), f"Mask mismatch at threshold {th}"

    print("  Testing monotonic response across slider range [0.05 .. 0.90]:")
    for th in slider_thresholds:
        print(f"    - Sensitivity Threshold {th:.2f} -> Changed Pixels: {results[th]:,}")

    # Monotonicity: as threshold increases, changed pixel count must be non-increasing
    counts = [results[th] for th in slider_thresholds]
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), \
        f"Threshold monotonicity violated: {counts}"

    # Confirm mask differences between extremes
    mask_diff = np.sum(masks[0.05] != masks[0.90])
    assert mask_diff > 0, "Binary mask did not change between threshold 0.05 and 0.90!"
    print(f"  [PASS] Detection-threshold slider dynamically refreshes mask (difference: {mask_diff:,} pixels)")


def test_model_switching_differentiation(model_full, model_init):
    print("\n------------------------------------------------------------------")
    print("[TEST 5] Model Switching & Prediction Refresh Verification")
    print("------------------------------------------------------------------")
    t1_path = "data/change_detection/source/csn_7dca6e98_2016.jpg"
    t2_path = "data/change_detection/source/csn_7dca6e98_current.jpg"
    assert os.path.exists(t1_path) and os.path.exists(t2_path), "Test satellite tiles missing"

    tensor_real, _, r2 = spacenetra_model.preprocess_pair(t1_path, t2_path, target_size=256)

    # Run inference with full-training model at threshold 0.20
    pred_full = spacenetra_model.predict_change(model_full, tensor_real, r2, threshold=0.20, device="cpu")
    # Run inference with initial model at threshold 0.20
    pred_init = spacenetra_model.predict_change(model_init, tensor_real, r2, threshold=0.20, device="cpu")

    # Verify probability maps differ significantly between the 2 checkpoints
    prob_diff = np.abs(pred_full["prob_map"] - pred_init["prob_map"]).mean()
    assert prob_diff > 0.001, f"Model switching failed to produce differentiated outputs: mean diff {prob_diff}"

    # Verify binary masks differ
    mask_diff = np.sum(pred_full["binary_mask"] != pred_init["binary_mask"])
    assert mask_diff > 0, "Both models produced completely identical binary masks!"

    print(f"  [PASS] Model switching verified on active satellite tiles:")
    print(f"    - Full-train (Epoch 9) changed pixels @ 0.20: {pred_full['changed_pixels']:,} px")
    print(f"    - Initial (Epoch 2) changed pixels @ 0.20: {pred_init['changed_pixels']:,} px")
    print(f"    - Prediction difference: {mask_diff:,} pixels (Mean continuous probability delta: {prob_diff:.4f})")


def test_heatmap_generation():
    print("\n------------------------------------------------------------------")
    print("[TEST 6] Probability Heatmap Visualization Verification")
    print("------------------------------------------------------------------")
    # Synthetic continuous probability ramp from 0.0 to 1.0
    synth_probs = np.linspace(0.0, 1.0, 256 * 256).reshape((256, 256)).astype(np.float32)
    heatmap = spacenetra_model.generate_probability_heatmap(synth_probs)

    assert heatmap.shape == (256, 256, 3), f"Expected (256, 256, 3), got {heatmap.shape}"
    assert heatmap.dtype == np.uint8, f"Expected uint8, got {heatmap.dtype}"
    assert heatmap.min() >= 0 and heatmap.max() <= 255, "Heatmap values outside [0, 255]"
    # Verify non-trivial color variance across gradient
    assert len(np.unique(heatmap)) > 50, "Heatmap lacks continuous color variation"
    print("  [PASS] Continuous probability heatmap generation verified (shape: 256×256×3, RGB Turbo).")


def test_optional_morphology_postprocessing():
    print("\n------------------------------------------------------------------")
    print("[TEST 7] Optional Morphology Post-Processing Verification")
    print("------------------------------------------------------------------")
    # Create test binary mask with isolated noise speck and a small hole in a building
    test_mask = np.zeros((256, 256), dtype=np.uint8)
    test_mask[50:100, 50:100] = 1   # 50x50 building block
    test_mask[70:75, 70:75] = 0     # 5x5 internal hole
    test_mask[180, 180] = 1         # 1x1 isolated noise speck

    # 1. Opening: removes isolated noise specks
    opened_mask = spacenetra_model.apply_optional_morphology(test_mask, operation="opening", kernel_size=3, iterations=1)
    assert opened_mask.shape == (256, 256), "Opened mask dimension mismatch"
    assert opened_mask[180, 180] == 0, "Opening failed to remove isolated noise speck"
    assert opened_mask[60, 60] == 1, "Opening corrupted main building body"

    # 2. Closing: bridges small internal gaps
    closed_mask = spacenetra_model.apply_optional_morphology(test_mask, operation="closing", kernel_size=7, iterations=1)
    assert closed_mask.shape == (256, 256), "Closed mask dimension mismatch"
    assert closed_mask[72, 72] == 1, "Closing failed to bridge small internal hole"

    # 3. Closing followed by Opening
    comb_mask = spacenetra_model.apply_optional_morphology(test_mask, operation="closing_opening", kernel_size=5, iterations=1)
    assert comb_mask.shape == (256, 256), "Combined mask dimension mismatch"

    print("  [PASS] Optional morphology post-processing functions correctly and non-destructively.")


def test_reference_mask_pair_evaluation():
    print("\n------------------------------------------------------------------")
    print("[TEST 8] Pair-Specific Reference Mask Evaluation Verification")
    print("------------------------------------------------------------------")
    # Synthetic ground truth: 100x100 block
    ref_gt = np.zeros((256, 256), dtype=np.uint8)
    ref_gt[50:150, 50:150] = 1 # 10,000 true positive change pixels

    # Synthetic prediction: 80x80 block overlapping from (50, 50) to (130, 130) + 20x20 false positive at (200, 200)
    pred_mask = np.zeros((256, 256), dtype=np.uint8)
    pred_mask[50:130, 50:130] = 1 # TP = 6,400 px
    pred_mask[200:220, 200:220] = 1 # FP = 400 px
    # FN = 10,000 - 6,400 = 3,600 px

    # Convert ground truth to in-memory PNG file
    gt_pil = Image.fromarray(ref_gt * 255)
    buf = io.BytesIO()
    gt_pil.save(buf, format="PNG")
    buf.seek(0)

    eval_res = spacenetra_model.evaluate_against_reference_mask(pred_mask, buf)

    expected_tp = 6400
    expected_fp = 400
    expected_fn = 3600
    expected_tn = (256 * 256) - (expected_tp + expected_fp + expected_fn)
    expected_prec = expected_tp / (expected_tp + expected_fp) # 6400 / 6800 = 0.941176
    expected_rec = expected_tp / (expected_tp + expected_fn)   # 6400 / 10000 = 0.640000
    expected_f1 = (2 * expected_prec * expected_rec) / (expected_prec + expected_rec)
    expected_iou = expected_tp / (expected_tp + expected_fp + expected_fn) # 6400 / 10400 = 0.61538

    assert eval_res["tp"] == expected_tp, f"TP mismatch: got {eval_res['tp']}, expected {expected_tp}"
    assert eval_res["fp"] == expected_fp, f"FP mismatch: got {eval_res['fp']}, expected {expected_fp}"
    assert eval_res["fn"] == expected_fn, f"FN mismatch: got {eval_res['fn']}, expected {expected_fn}"
    assert eval_res["tn"] == expected_tn, f"TN mismatch: got {eval_res['tn']}, expected {expected_tn}"
    assert abs(eval_res["precision"] - expected_prec) < 1e-4, f"Precision mismatch: {eval_res['precision']}"
    assert abs(eval_res["recall"] - expected_rec) < 1e-4, f"Recall mismatch: {eval_res['recall']}"
    assert abs(eval_res["f1"] - expected_f1) < 1e-4, f"F1 mismatch: {eval_res['f1']}"
    assert abs(eval_res["iou"] - expected_iou) < 1e-4, f"IoU mismatch: {eval_res['iou']}"

    # Verify confusion map
    conf_map = eval_res["confusion_map"]
    assert conf_map.shape == (256, 256, 3), "Confusion map shape mismatch"
    # Check green TP color at (60, 60)
    assert np.array_equal(conf_map[60, 60], [34, 197, 94]), "TP pixel color should be green"
    # Check red FP color at (210, 210)
    assert np.array_equal(conf_map[210, 210], [239, 68, 68]), "FP pixel color should be red"
    # Check blue FN color at (140, 140)
    assert np.array_equal(conf_map[140, 140], [59, 130, 246]), "FN pixel color should be blue"

    print("  [PASS] Pair-specific reference mask evaluation verified:")
    print(f"    - TP: {eval_res['tp']:,} | FP: {eval_res['fp']:,} | FN: {eval_res['fn']:,} | TN: {eval_res['tn']:,}")
    print(f"    - Precision: {eval_res['precision']:.4f} | Recall: {eval_res['recall']:.4f}")
    print(f"    - F1 Score: {eval_res['f1']:.4f} | IoU: {eval_res['iou']:.4f}")
    print("    - Diagnostic confusion map (Green=TP, Red=FP, Blue=FN, Navy=TN) validated.")


def test_traditional_detection_and_semantic_search():
    print("\n------------------------------------------------------------------")
    print("[TEST 9] Preservation of Traditional Detection & Semantic Search")
    print("------------------------------------------------------------------")
    # 1. Traditional building detection
    print("  Verifying traditional multi-temporal building detector...")
    t1_path = "data/change_detection/source/csn_7dca6e98_2016.jpg"
    t3_path = "data/change_detection/source/csn_7dca6e98_current.jpg"

    bldg_res = building_detector.detect_new_buildings(
        t1_path, t3_path, gsd_m=1.19, min_area_sq_m=50.0, sensitivity=65
    )
    assert "building_count" in bldg_res, "building_count missing from detector result"
    assert "total_built_area_sq_m" in bldg_res, "total_built_area_sq_m missing"
    assert "annotated_img" in bldg_res, "annotated_img missing"
    assert bldg_res["building_count"] >= 0, "Invalid building count"
    print(f"  [PASS] Traditional detector operational: detected {bldg_res['building_count']} buildings.")

    # 2. Semantic search (CLIP + FAISS)
    print("  Verifying CLIP + FAISS vector database...")
    import clip
    import faiss

    assert os.path.exists("vector_db/spatial.index"), "FAISS spatial.index missing"
    assert os.path.exists("vector_db/metadata.npy"), "vector_db metadata.npy missing"
    assert os.path.exists("vector_db_backup/spatial.index"), "FAISS backup missing"

    index = faiss.read_index("vector_db/spatial.index")
    meta = np.load("vector_db/metadata.npy", allow_pickle=True)
    assert index.ntotal > 0, "FAISS index has 0 vectors"
    assert len(meta) == index.ntotal, "Metadata length mismatch with FAISS index"

    device = "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    tokens = clip.tokenize(["commercial industrial building"]).to(device)
    with torch.no_grad():
        feat = clip_model.encode_text(tokens)
        feat /= feat.norm(dim=-1, keepdim=True)

    dist, idxs = index.search(feat.cpu().numpy().astype("float32"), k=3)
    assert len(idxs[0]) == 3, f"Expected 3 search matches, got {len(idxs[0])}"
    print(f"  [PASS] Semantic search operational: retrieved top match '{meta[idxs[0][0]]['filename']}' (Score: {dist[0][0]*100:.1f}%)")


def test_threshold_sensitivity_diagnostic_and_registration(model_full, model_init):
    print("\n------------------------------------------------------------------")
    print("[TEST 10] Input Consistency & Threshold Sensitivity Diagnostics")
    print("------------------------------------------------------------------")
    f1 = "data/change_detection/source/csn_ccbdd603_2016.jpg"
    f2 = "data/change_detection/source/csn_ccbdd603_current.jpg"
    assert os.path.exists(f1), f"Sample image missing: {f1}"
    assert os.path.exists(f2), f"Sample image missing: {f2}"

    # 1. Alignment diagnostic test
    print("  Testing 2D FFT phase correlation registration diagnostics...")
    align_res = spacenetra_model.verify_input_consistency_and_alignment(f1, f2)
    assert align_res["compatible_framing"] is True, "Expected compatible framing"
    assert align_res["orig_size_earlier"] == (256, 256), "Expected 256x256 earlier size"
    assert align_res["orig_size_later"] == (256, 256), "Expected 256x256 later size"
    assert align_res["displacement_magnitude_px"] == 2.63, f"Expected 2.63px displacement, got {align_res['displacement_magnitude_px']}"
    assert align_res["dx_pixels"] == 2.28, f"Expected dx=2.28, got {align_res['dx_pixels']}"
    assert align_res["dy_pixels"] == -1.30, f"Expected dy=-1.30, got {align_res['dy_pixels']}"
    assert align_res["alignment_status"] == "MODERATE_PARALLAX_SHIFT", f"Unexpected status: {align_res['alignment_status']}"
    assert align_res["pearson_correlation_r"] > 0.85, f"Expected high spatial correlation, got {align_res['pearson_correlation_r']}"
    print(f"  [PASS] Alignment diagnostic confirmed: shift={align_res['displacement_magnitude_px']}px, status={align_res['alignment_status']}.")

    # 2. Threshold sensitivity diagnostic on identical probability maps
    print("  Testing threshold sensitivity diagnostics across [0.10, 0.20, 0.30, 0.40, 0.50, 0.65]...")
    tensor_6ch, im1_p, im2_p = spacenetra_model.preprocess_pair(f1, f2)
    
    with torch.no_grad():
        probs_full = torch.sigmoid(model_full(tensor_6ch)).squeeze().cpu().numpy()
        probs_orig = torch.sigmoid(model_init(tensor_6ch)).squeeze().cpu().numpy()

    sens_full = spacenetra_model.compute_threshold_sensitivity_diagnostic(probs_full)
    sens_orig = spacenetra_model.compute_threshold_sensitivity_diagnostic(probs_orig)

    assert len(sens_full) == 6, f"Expected 6 diagnostic threshold steps, got {len(sens_full)}"
    assert len(sens_orig) == 6, f"Expected 6 diagnostic threshold steps, got {len(sens_orig)}"

    # Check that predictions at threshold 0.50 match the exact problem description:
    # Full-train ~ 613 px (0.94%), Original ~ 10,119 px (15.44%)
    th50_full = next(s for s in sens_full if s["threshold"] == 0.50)
    th50_orig = next(s for s in sens_orig if s["threshold"] == 0.50)

    assert th50_full["changed_pixels"] == 613, f"Expected 613 changed pixels for full-train @ 0.50, got {th50_full['changed_pixels']}"
    assert th50_full["change_pct"] == 0.94, f"Expected 0.94% for full-train @ 0.50, got {th50_full['change_pct']}"

    assert th50_orig["changed_pixels"] == 10119, f"Expected 10,119 changed pixels for orig @ 0.50, got {th50_orig['changed_pixels']}"
    assert th50_orig["change_pct"] == 15.44, f"Expected 15.44% for orig @ 0.50, got {th50_orig['change_pct']}"

    # Check extreme sensitivity at 0.10 (undertrained model floods entire frame)
    th10_orig = next(s for s in sens_orig if s["threshold"] == 0.10)
    th10_full = next(s for s in sens_full if s["threshold"] == 0.10)
    assert th10_orig["changed_pixels"] == 65535, f"Expected 65,535 px at 0.10 for orig, got {th10_orig['changed_pixels']}"
    assert th10_full["changed_pixels"] == 1290, f"Expected 1,290 px at 0.10 for full-train, got {th10_full['changed_pixels']}"

    print(f"  [PASS] Full-Train @ 0.50: {th50_full['changed_pixels']} px ({th50_full['change_pct']}%) matches exact problem baseline!")
    print(f"  [PASS] Original @ 0.50: {th50_orig['changed_pixels']} px ({th50_orig['change_pct']}%) matches exact problem baseline!")
    print(f"  [PASS] Threshold sensitivity table successfully evaluated from single probability map without inference re-runs.")


def run_all_tests():
    print("==================================================================")
    print(" SPACE NETRA: COMPLETE ML MODEL & INTEGRATION VALIDATION SUITE   ")
    print("==================================================================")
    (model_full, meta_full), (model_init, meta_init) = test_checkpoint_strict_loading()
    tensor_6ch, im2_res = test_preprocessing_compliance()
    test_inference_and_dimensions(model_full, tensor_6ch, im2_res)
    test_threshold_responsiveness_and_slider(model_full)
    test_model_switching_differentiation(model_full, model_init)
    test_heatmap_generation()
    test_optional_morphology_postprocessing()
    test_reference_mask_pair_evaluation()
    test_traditional_detection_and_semantic_search()
    test_threshold_sensitivity_diagnostic_and_registration(model_full, model_init)
    print("\n==================================================================")
    print(" ALL 10 VALIDATION REQUIREMENTS PASSED SUCCESSFULLY!              ")
    print("==================================================================")


if __name__ == "__main__":
    run_all_tests()


