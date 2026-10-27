import spacenetra_model
import torch
from PIL import Image

f1 = 'data/change_detection/source/csn_ccbdd603_2016.jpg'
f2 = 'data/change_detection/source/csn_ccbdd603_current.jpg'

diag = spacenetra_model.verify_input_consistency_and_alignment(f1, f2)
print("Alignment diagnostic:")
for k, v in diag.items():
    print(f"  {k}: {v}")

tensor_6ch, im1_p, im2_p = spacenetra_model.preprocess_pair(f1, f2)

model_full, _, _ = spacenetra_model.load_model(spacenetra_model.FULLTRAIN_CHECKPOINT_PATH)
with torch.no_grad():
    probs_full = torch.sigmoid(model_full(tensor_6ch)).squeeze().cpu().numpy()

print("\n--- Full-Train (Epoch 9) Threshold Sensitivity ---")
sens_full = spacenetra_model.compute_threshold_sensitivity_diagnostic(probs_full)
for s in sens_full:
    print(f"  Threshold {s['threshold']:.2f}: {s['changed_pixels']:,} px ({s['change_pct']}%)")

model_orig, _, _ = spacenetra_model.load_model(spacenetra_model.INITIAL_CHECKPOINT_PATH)
with torch.no_grad():
    probs_orig = torch.sigmoid(model_orig(tensor_6ch)).squeeze().cpu().numpy()

print("\n--- Original (Epoch 2) Threshold Sensitivity ---")
sens_orig = spacenetra_model.compute_threshold_sensitivity_diagnostic(probs_orig)
for s in sens_orig:
    print(f"  Threshold {s['threshold']:.2f}: {s['changed_pixels']:,} px ({s['change_pct']}%)")
