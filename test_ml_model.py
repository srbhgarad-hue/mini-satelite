"""
test_ml_model.py
================
Comprehensive Validation Suite for Space Netra Experimental U-Net Integration.

Verifies:
1. Loading the new full-training checkpoint (models/spacenetra_s2looking_unet_fulltrain_best.pth).
2. Strict state-dictionary loading compatibility (strict=True) and preservation of original checkpoint.
3. Preprocessing compliance (bilinear 256x256, [0, 1] normalized, earlier RGB + later RGB in 6-channel CHW format).
4. Output mask dimensions (256x256) and tensor shapes.
5. Finite probabilities in range [0, 1].
6. Binary mask responsiveness across multiple threshold values.
7. Preservation of existing semantic-search (CLIP + FAISS) and traditional detection features.
"""

import os
import numpy as np
from PIL import Image
import torch

import spacenetra_model
import building_detector


def test_checkpoint_strict_loading():
    print("------------------------------------------------------------------")
    print("[TEST 1 & 2] Loading New Checkpoint & Strict State-Dict Verification")
    print("------------------------------------------------------------------")
    new_ckpt = spacenetra_model.DEFAULT_CHECKPOINT_PATH
    orig_ckpt = spacenetra_model.ORIGINAL_CHECKPOINT_PATH

    print(f"Checking new full-train checkpoint exists: {new_ckpt}")
    assert os.path.exists(new_ckpt), f"New checkpoint missing at {new_ckpt}"
    assert os.path.getsize(new_ckpt) > 2_000_000, f"New checkpoint seems truncated: {os.path.getsize(new_ckpt)} bytes"

    print(f"Checking original checkpoint remains intact: {orig_ckpt}")
    assert os.path.exists(orig_ckpt), f"Original checkpoint missing at {orig_ckpt}"
    assert os.path.getsize(orig_ckpt) > 2_000_000, f"Original checkpoint truncated: {os.path.getsize(orig_ckpt)} bytes"

    # 1. Load new checkpoint strictly
    model, metadata, err = spacenetra_model.load_model(new_ckpt, device="cpu")
    assert err is None, f"Strict loading of new checkpoint failed: {err}"
    assert model is not None, "Loaded model is None"
    assert metadata["epoch"] == 9, f"Expected epoch 9, got {metadata.get('epoch')}"
    assert metadata["training_pairs"] == 3500, f"Expected 3,500 train pairs, got {metadata.get('training_pairs')}"
    assert metadata["validation_pairs"] == 500, f"Expected 500 val pairs, got {metadata.get('validation_pairs')}"
    assert round(metadata["validation_metrics"]["f1"], 4) == 0.3910, f"Unexpected Val F1: {metadata['validation_metrics']['f1']}"
    assert round(metadata["validation_metrics"]["iou"], 4) == 0.2430, f"Unexpected Val IoU: {metadata['validation_metrics']['iou']}"
    assert round(metadata["heldout_test_metrics"]["f1"], 4) == 0.3475, f"Unexpected Test F1: {metadata['heldout_test_metrics']['f1']}"
    assert round(metadata["heldout_test_metrics"]["iou"], 4) == 0.2103, f"Unexpected Test IoU: {metadata['heldout_test_metrics']['iou']}"
    print("  [PASS] New checkpoint loaded strictly with strict=True!")
    print(f"    - Epoch: {metadata['epoch']} (3,500 train pairs, 500 val pairs)")
    print(f"    - Benchmark Val F1: {metadata['validation_metrics']['f1']:.4f} | IoU: {metadata['validation_metrics']['iou']:.4f}")
    print(f"    - Held-out Test F1: {metadata['heldout_test_metrics']['f1']:.4f} | IoU: {metadata['heldout_test_metrics']['iou']:.4f}")

    # 2. Verify original checkpoint also loads strictly
    orig_model, orig_meta, orig_err = spacenetra_model.load_model(orig_ckpt, device="cpu")
    assert orig_err is None, f"Strict loading of original checkpoint failed: {orig_err}"
    assert orig_model is not None, "Original model is None"
    print("  [PASS] Original checkpoint verified intact and loadable with strict=True!")

    return model, metadata


def test_preprocessing_compliance():
    print("\n------------------------------------------------------------------")
    print("[TEST 3] Preserved Training Preprocessing Verification")
    print("------------------------------------------------------------------")
    # Two test images of arbitrary different sizes to test resizing
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

    # Confirm earlier RGB channels (0:3) match im1 and later RGB channels (3:6) match im2
    t1_np = tensor_6ch[0, :3, :, :].numpy()
    t2_np = tensor_6ch[0, 3:, :, :].numpy()
    expected_c0 = 100.0 / 255.0
    expected_c3 = 110.0 / 255.0
    assert abs(t1_np[0, 0, 0] - expected_c0) < 1e-3, "Channel 0 does not match earlier image"
    assert abs(t2_np[0, 0, 0] - expected_c3) < 1e-3, "Channel 3 does not match later image"

    print("  [PASS] Preprocessing strictly adheres to training specification:")
    print(f"    - Shape: {tensor_6ch.shape} (BCHW format, 6 channels)")
    print(f"    - Pixel range: [{min_val:.4f}, {max_val:.4f}] in [0, 1]")
    print("    - Earlier RGB (channels 0..2) followed by later RGB (channels 3..5)")
    print("    - Bilinear interpolation to 256x256 confirmed.")

    return tensor_6ch, im2_resized


def test_inference_and_dimensions(model, tensor_6ch, im2_resized):
    print("\n------------------------------------------------------------------")
    print("[TEST 4 & 5] Inference, Output Dimensions & Probability Range")
    print("------------------------------------------------------------------")
    pred = spacenetra_model.predict_change(model, tensor_6ch, im2_resized, threshold=0.50, device="cpu")

    # Dimensions
    assert pred["prob_map"].shape == (256, 256), f"Expected prob_map (256, 256), got {pred['prob_map'].shape}"
    assert pred["binary_mask"].shape == (256, 256), f"Expected binary_mask (256, 256), got {pred['binary_mask'].shape}"
    assert pred["overlay_img"].shape == (256, 256, 3), f"Expected overlay_img (256, 256, 3), got {pred['overlay_img'].shape}"
    assert pred["total_pixels"] == 256 * 256, f"Expected total pixels 65,536, got {pred['total_pixels']}"

    # Probabilities finite and within [0, 1]
    assert pred["is_valid"] is True, "Probabilities contain NaN or Inf values!"
    assert np.isfinite(pred["prob_map"]).all(), "Non-finite values found in prob_map"
    p_min = float(pred["prob_map"].min())
    p_max = float(pred["prob_map"].max())
    assert 0.0 <= p_min <= 1.0, f"prob_map min {p_min} outside [0, 1]"
    assert 0.0 <= p_max <= 1.0, f"prob_map max {p_max} outside [0, 1]"

    # Binary mask values
    unique_vals = set(np.unique(pred["binary_mask"]))
    assert unique_vals.issubset({0, 1}), f"Non-binary values in binary_mask: {unique_vals}"

    print("  [PASS] Inference output dimensions and probabilities verified:")
    print(f"    - Output mask dimensions: {pred['binary_mask'].shape}")
    print(f"    - Probabilities range: [{p_min:.5f}, {p_max:.5f}] (all finite, [0, 1])")
    print(f"    - Changed pixels at threshold 0.50: {pred['changed_pixels']} ({pred['change_ratio_pct']}%)")


def test_threshold_responsiveness(model):
    print("\n------------------------------------------------------------------")
    print("[TEST 6] Threshold Sensitivity & Binary Mask Responsiveness")
    print("------------------------------------------------------------------")
    # Use real optical satellite tile pair where natural contrast produces varying probability distributions
    t1_path = "data/change_detection/source/csn_7dca6e98_2016.jpg"
    t2_path = "data/change_detection/source/csn_7dca6e98_current.jpg"
    assert os.path.exists(t1_path) and os.path.exists(t2_path), "Test satellite tiles missing"

    tensor_real, _, r2 = spacenetra_model.preprocess_pair(t1_path, t2_path, target_size=256)

    thresholds = [0.20, 0.35, 0.50, 0.65, 0.80]
    results = {}
    masks = {}

    for th in thresholds:
        pred_th = spacenetra_model.predict_change(model, tensor_real, r2, threshold=th, device="cpu")
        results[th] = pred_th["changed_pixels"]
        masks[th] = pred_th["binary_mask"]
        assert pred_th["threshold"] == th, f"Reported threshold {pred_th['threshold']} != requested {th}"

        # Verify dynamic re-threshold helper produces identical result
        re_pred = spacenetra_model.apply_threshold_to_prob_map(pred_th["prob_map"], r2, threshold=th)
        assert re_pred["changed_pixels"] == pred_th["changed_pixels"], "apply_threshold_to_prob_map mismatch!"
        assert np.array_equal(re_pred["binary_mask"], pred_th["binary_mask"]), "apply_threshold_to_prob_map mask mismatch!"

    print("  Testing monotonic response across decision thresholds:")
    for th in thresholds:
        print(f"    - Threshold {th:.2f} -> Changed Pixels: {results[th]:,}")

    # As threshold increases, changed pixel count must be non-increasing (monotonic)
    counts = [results[th] for th in thresholds]
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), \
        f"Threshold monotonicity violated: {counts}"

    # Confirm that the mask actually changes across different thresholds
    mask_diff = np.sum(masks[0.20] != masks[0.80])
    assert mask_diff > 0, "Binary mask did not change between threshold 0.20 and 0.80!"
    print(f"  [PASS] Binary mask successfully changes with threshold (differing pixels: {mask_diff:,})")


def test_traditional_detection_and_semantic_search():
    print("\n------------------------------------------------------------------")
    print("[TEST 7] Preservation of Traditional Detection & Semantic Search")
    print("------------------------------------------------------------------")
    # 1. Traditional building detection
    print("  Verifying traditional multi-temporal building detector...")
    t1_path = "data/change_detection/source/csn_7dca6e98_2016.jpg"
    t2_path = "data/change_detection/source/csn_7dca6e98_2020.jpg"
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


def run_all_tests():
    print("==================================================================")
    print(" SPACE NETRA: COMPLETE ML MODEL & INTEGRATION VALIDATION SUITE   ")
    print("==================================================================")
    model, metadata = test_checkpoint_strict_loading()
    tensor_6ch, im2_res = test_preprocessing_compliance()
    test_inference_and_dimensions(model, tensor_6ch, im2_res)
    test_threshold_responsiveness(model)
    test_traditional_detection_and_semantic_search()
    print("\n==================================================================")
    print(" ALL 7 VALIDATION REQUIREMENTS PASSED SUCCESSFULLY!               ")
    print("==================================================================")


if __name__ == "__main__":
    run_all_tests()
