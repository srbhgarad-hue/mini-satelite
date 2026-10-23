"""
test_ml_model.py
================
Validation script for spacenetra_model.py and checkpoint loading.
"""

import os
import numpy as np
from PIL import Image
import torch
import spacenetra_model

def run_ml_tests():
    print("==================================================================")
    print(" SPACE NETRA: S2LOOKING EXPERIMENTAL U-NET VALIDATION SUITE       ")
    print("==================================================================")

    # 1. Checkpoint Strict Loading Test
    ckpt_path = "models/spacenetra_s2looking_unet.pth"
    print(f"[TEST 1] Loading checkpoint from: {ckpt_path}")
    model, metadata, err = spacenetra_model.load_model(ckpt_path, device="cpu")
    
    assert err is None, f"Model loading failed: {err}"
    assert model is not None, "Model is None after loading"
    print("[PASS] Checkpoint loaded with strict=True!")
    print(f"  - Epoch: {metadata['epoch']}")
    print(f"  - Input Channels: {metadata['input_channels']}")
    print(f"  - Image Size: {metadata['image_size']}")
    print(f"  - Val Metrics: F1={metadata['validation_metrics']['f1']:.4f}, IoU={metadata['validation_metrics']['iou']:.4f}")

    # 2. Preprocessing Test
    print("\n[TEST 2] Testing Training-Identical Preprocessing...")
    dummy1 = Image.new("RGB", (300, 400), color=(120, 150, 180))
    dummy2 = Image.new("RGB", (500, 350), color=(140, 170, 200))
    tensor_6ch, im1_res, im2_res = spacenetra_model.preprocess_pair(dummy1, dummy2, target_size=256)

    assert tensor_6ch.shape == (1, 6, 256, 256), f"Unexpected tensor shape: {tensor_6ch.shape}"
    assert tensor_6ch.dtype == torch.float32, f"Unexpected tensor dtype: {tensor_6ch.dtype}"
    assert 0.0 <= tensor_6ch.min().item() <= 1.0, "Tensor values not in [0, 1]"
    assert 0.0 <= tensor_6ch.max().item() <= 1.0, "Tensor values not in [0, 1]"
    print(f"[PASS] Preprocessing produced 6-channel tensor: {tensor_6ch.shape} in range [0, 1]")

    # 3. Inference Test
    print("\n[TEST 3] Running Inference...")
    pred = spacenetra_model.predict_change(model, tensor_6ch, im2_res, threshold=0.50, device="cpu")

    assert pred["prob_map"].shape == (256, 256), f"Unexpected prob_map shape: {pred['prob_map'].shape}"
    assert pred["binary_mask"].shape == (256, 256), f"Unexpected binary_mask shape: {pred['binary_mask'].shape}"
    assert pred["overlay_img"].shape == (256, 256, 3), f"Unexpected overlay_img shape: {pred['overlay_img'].shape}"
    assert pred["is_valid"] is True, "Inference output contains NaN or Inf values!"
    assert set(np.unique(pred["binary_mask"])).issubset({0, 1}), "Binary mask has non-binary values"
    print("[PASS] Inference succeeded!")
    print(f"  - Prob Map Range: [{pred['prob_map'].min():.4f}, {pred['prob_map'].max():.4f}]")
    print(f"  - Changed Pixels: {pred['changed_pixels']} ({pred['change_ratio_pct']}%)")
    print(f"  - Finite Values Verified: {pred['is_valid']}")

    # 4. Real Image Pair Test (using verified real cached source tiles)
    t1_path = "data/change_detection/source/csn_7dca6e98_2016.jpg"
    t2_path = "data/change_detection/source/csn_7dca6e98_current.jpg"
    if os.path.exists(t1_path) and os.path.exists(t2_path):
        print(f"\n[TEST 4] Testing with verified real satellite pair ({t1_path})...")
        t_real, r1, r2 = spacenetra_model.preprocess_pair(t1_path, t2_path, target_size=256)
        pred_real = spacenetra_model.predict_change(model, t_real, r2, threshold=0.50, device="cpu")
        print(f"[PASS] Real image pair test passed! Changed pixels: {pred_real['changed_pixels']}")

    print("\n==================================================================")
    print(" ALL EXPERIMENTAL U-NET TESTS PASSED SUCCESSFULLY!                ")
    print("==================================================================")

if __name__ == "__main__":
    run_ml_tests()
