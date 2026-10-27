"""
spacenetra_model.py
===================
Experimental Deep Learning Change Detection Model for Space Netra.
Architecture: 6-channel Bitemporal U-Net trained on S2Looking subset in Google Colab.
Input: Two RGB images (Earlier + Later) resized to 256x256, concatenated along channel dimension.
Output: Single-channel binary change probability map.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn

INITIAL_CHECKPOINT_PATH = os.path.join("models", "spacenetra_s2looking_unet.pth")
FULLTRAIN_CHECKPOINT_PATH = os.path.join("models", "spacenetra_s2looking_unet_fulltrain_best.pth")

# Backward compatibility aliases
ORIGINAL_CHECKPOINT_PATH = INITIAL_CHECKPOINT_PATH
DEFAULT_CHECKPOINT_PATH = FULLTRAIN_CHECKPOINT_PATH

CHECKPOINTS_CONFIG: Dict[str, Dict] = {
    "fulltrain": {
        "id": "fulltrain",
        "title": "Full-training model — trained on 3,500 pairs, best checkpoint from epoch 9",
        "short_label": "Full-Training Model (3,500 pairs, Epoch 9 best)",
        "path": FULLTRAIN_CHECKPOINT_PATH,
        "epoch": 9,
        "training_pairs": 3500,
        "validation_pairs": 500,
        "description": "Trained in Google Colab on all 3,500 training pairs of the S2Looking benchmark dataset. Model selection evaluated on 500 validation pairs and final evaluation on 1,000 held-out test pairs.",
        "metrics_val": {"precision": 0.4403, "recall": 0.3516, "f1": 0.3910, "iou": 0.2430, "loss": 0.7941},
        "metrics_test": {"test_pairs": 1000, "precision": 0.3829, "recall": 0.3181, "f1": 0.3475, "iou": 0.2103}
    },
    "initial": {
        "id": "initial",
        "title": "Initial model — trained on 500 S2Looking training pairs",
        "short_label": "Initial Model (500 pairs, Epoch 2)",
        "path": INITIAL_CHECKPOINT_PATH,
        "epoch": 2,
        "training_pairs": 500,
        "validation_pairs": 100,
        "description": "Trained in Google Colab on a sampled 500-pair subset of the S2Looking benchmark dataset. Model selection evaluated on 100 validation pairs.",
        "metrics_val": {"precision": 0.2993, "recall": 0.1534, "f1": 0.2028, "iou": 0.1129, "loss": 1.1263},
        "metrics_test": None
    }
}


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3x3 convolutions with BatchNorm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class SpaceNetraUNet(nn.Module):
    """
    Exact U-Net architecture used during S2Looking experimental training.
    Encoder: 6 -> 16 -> 32 -> 64 -> 128
    Middle: 128 -> 256
    Decoder: 256 -> 128 -> 64 -> 32 -> 16
    Output: 16 -> 1 (binary change logits)
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = conv_block(6, 16)
        self.enc2 = conv_block(16, 32)
        self.enc3 = conv_block(32, 64)
        self.enc4 = conv_block(64, 128)

        self.middle = conv_block(128, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = conv_block(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = conv_block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = conv_block(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = conv_block(32, 16)

        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        middle = self.middle(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(middle), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.output(d1)


def load_model(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    device: Optional[str] = None
) -> Tuple[Optional[SpaceNetraUNet], Optional[Dict], Optional[str]]:
    """
    Loads the trained SpaceNetraUNet checkpoint strictly.
    Returns: (model, checkpoint_metadata, error_message)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(checkpoint_path):
        return None, None, f"Checkpoint file not found at project path: '{checkpoint_path}'"

    try:
        model = SpaceNetraUNet()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if "model_state_dict" not in checkpoint:
            return None, None, "Checkpoint missing 'model_state_dict' key."

        # Load weights with strict=True as mandated
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        model.eval()

        ckpt_epoch = checkpoint.get("epoch", 2 if "fulltrain" not in checkpoint_path else 9)
        training_pairs = checkpoint.get("training_pairs", 500 if ckpt_epoch == 2 else 3500)
        validation_pairs = checkpoint.get("validation_pairs", 100 if ckpt_epoch == 2 else 500)
        default_val = (
            {"precision": 0.2993, "recall": 0.1534, "f1": 0.2028, "iou": 0.1129, "loss": 1.1263}
            if ckpt_epoch == 2 else
            {"precision": 0.4403, "recall": 0.3516, "f1": 0.3910, "iou": 0.2430, "loss": 0.7941}
        )
        validation_metrics = checkpoint.get("validation_metrics", default_val)

        heldout_test_metrics = None
        if ckpt_epoch == 9 or "fulltrain" in checkpoint_path:
            heldout_test_metrics = {
                "test_pairs": 1000,
                "precision": 0.3829,
                "recall": 0.3181,
                "f1": 0.3475,
                "iou": 0.2103
            }

        metadata = {
            "checkpoint_path": checkpoint_path,
            "checkpoint_filename": os.path.basename(checkpoint_path),
            "epoch": ckpt_epoch,
            "image_size": checkpoint.get("image_size", 256),
            "input_channels": checkpoint.get("input_channels", 6),
            "output_channels": checkpoint.get("output_channels", 1),
            "training_pairs": training_pairs,
            "validation_pairs": validation_pairs,
            "validation_metrics": validation_metrics,
            "heldout_test_metrics": heldout_test_metrics,
            "is_experimental": True,
            "device": device
        }

        return model, metadata, None

    except Exception as e:
        return None, None, f"Failed to load checkpoint strictly: {e}"


def preprocess_pair(
    earlier_image: Union[str, Image.Image, np.ndarray],
    later_image: Union[str, Image.Image, np.ndarray],
    target_size: int = 256
) -> Tuple[torch.Tensor, Image.Image, Image.Image]:
    """
    Reproduces the exact training preprocessing:
    1. Load earlier and later images.
    2. Convert both images to RGB.
    3. Resize each image to 256x256 using bilinear interpolation.
    4. Convert pixel values to float32 in range [0, 1].
    5. Concatenate earlier RGB + later RGB -> 6 channels.
    6. Convert to (1, 6, 256, 256) float32 tensor.
    """
    # 1 & 2. Load and convert to RGB
    if isinstance(earlier_image, str):
        im1 = Image.open(earlier_image).convert("RGB")
    elif isinstance(earlier_image, np.ndarray):
        im1 = Image.fromarray(earlier_image).convert("RGB")
    else:
        im1 = earlier_image.convert("RGB")

    if isinstance(later_image, str):
        im2 = Image.open(later_image).convert("RGB")
    elif isinstance(later_image, np.ndarray):
        im2 = Image.fromarray(later_image).convert("RGB")
    else:
        im2 = later_image.convert("RGB")

    # 3. Resize to target size with bilinear interpolation
    im1_resized = im1.resize((target_size, target_size), Image.Resampling.BILINEAR)
    im2_resized = im2.resize((target_size, target_size), Image.Resampling.BILINEAR)

    # 4. Convert to float32 in range [0, 1]
    arr1 = np.asarray(im1_resized, dtype=np.float32) / 255.0
    arr2 = np.asarray(im2_resized, dtype=np.float32) / 255.0

    # 5. Concatenate earlier RGB followed by later RGB (6 channels)
    combined = np.concatenate([arr1, arr2], axis=2)  # Shape: (256, 256, 6)

    # 6. Convert to CHW tensor: (1, 6, 256, 256)
    tensor = torch.from_numpy(combined.transpose(2, 0, 1).copy()).unsqueeze(0)

    return tensor, im1_resized, im2_resized


def generate_probability_heatmap(prob_map: np.ndarray, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """
    Converts a continuous float32 probability map [0.0, 1.0] into a vivid RGB heatmap visualization.
    Preserves continuous confidence values before thresholding.
    """
    scaled = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(scaled, colormap)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def apply_threshold_to_prob_map(
    prob_map: np.ndarray,
    later_pil: Image.Image,
    threshold: float = 0.50
) -> Dict:
    """
    Applies a decision threshold to an existing probability map without re-running model forward pass.
    Ensures that UI threshold adjustments and binary mask generation are always strictly identical.
    """
    binary_mask = (prob_map >= threshold).astype(np.uint8)

    # Create visual overlay: highlight predicted change in Coral/Amber (#F59E0B / #EF4444)
    # over the later RGB image
    later_arr = np.asarray(later_pil).copy()
    overlay = later_arr.copy()

    # Coral Red/Amber mask color (R=245, G=80, B=60)
    change_pixels = binary_mask > 0
    overlay[change_pixels] = (
        0.45 * overlay[change_pixels] + 0.55 * np.array([245, 80, 60], dtype=np.uint8)
    ).astype(np.uint8)

    heatmap_rgb = generate_probability_heatmap(prob_map)

    changed_count = int(np.sum(binary_mask))
    total_pixels = int(binary_mask.size)
    change_ratio_pct = round((changed_count / max(1, total_pixels)) * 100.0, 2)

    return {
        "prob_map": prob_map,
        "prob_heatmap": heatmap_rgb,
        "binary_mask": binary_mask,
        "overlay_img": overlay,
        "changed_pixels": changed_count,
        "total_pixels": total_pixels,
        "change_ratio_pct": change_ratio_pct,
        "threshold": threshold,
        "is_valid": bool(np.isfinite(prob_map).all())
    }


def predict_change(
    model: SpaceNetraUNet,
    tensor_6ch: torch.Tensor,
    later_pil: Image.Image,
    threshold: float = 0.50,
    device: str = "cpu"
) -> Dict:
    """
    Executes inference with the trained U-Net:
    1. Runs model forward pass.
    2. Applies sigmoid to output logits.
    3. Creates binary mask using threshold (default 0.50).
    4. Generates a highlighted overlay on the later image.
    """
    model.eval()
    tensor_input = tensor_6ch.to(device)

    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.sigmoid(logits)

    prob_map = probs.squeeze().cpu().numpy().astype(np.float32)
    return apply_threshold_to_prob_map(prob_map, later_pil, threshold=threshold)


def apply_optional_morphology(
    binary_mask: np.ndarray,
    operation: str = "opening",
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Optional post-processing for binary change mask.
    Operations:
      - 'opening': cv2.MORPH_OPEN (removes small isolated salt noise specks)
      - 'closing': cv2.MORPH_CLOSE (bridges small gaps/holes within contiguous footprints)
      - 'closing_opening': Closing followed by Opening
    Disabled by default. Does not alter raw probability maps.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask_u8 = (binary_mask * 255).astype(np.uint8)

    if operation == "opening":
        processed = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "closing":
        processed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == "closing_opening":
        closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        processed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)
    else:
        processed = mask_u8

    return (processed > 0).astype(np.uint8)


def evaluate_against_reference_mask(
    pred_binary: np.ndarray,
    ref_mask_input: Union[str, Image.Image, np.ndarray],
    target_size: int = 256
) -> Dict:
    """
    Evaluates binary prediction mask against a ground-truth reference mask for that specific image pair.
    Calculates precision, recall, F1, IoU, and generates a visual confusion diagnostic map.
    """
    if isinstance(ref_mask_input, str):
        ref_img = Image.open(ref_mask_input)
    elif isinstance(ref_mask_input, np.ndarray):
        ref_img = Image.fromarray(ref_mask_input)
    elif isinstance(ref_mask_input, Image.Image):
        ref_img = ref_mask_input
    else:
        # File-like object (BytesIO, Streamlit UploadedFile, etc.)
        ref_img = Image.open(ref_mask_input)

    # Convert to grayscale and resize using NEAREST to preserve discrete binary edges
    ref_gray = ref_img.convert("L").resize((target_size, target_size), Image.Resampling.NEAREST)
    ref_arr = np.asarray(ref_gray)

    # Threshold > 127 is change (positive), <= 127 is no change (negative)
    ref_b = (ref_arr > 127).astype(np.uint8)
    pred_b = (pred_binary > 0).astype(np.uint8)

    tp = int(np.sum((pred_b == 1) & (ref_b == 1)))
    fp = int(np.sum((pred_b == 1) & (ref_b == 0)))
    fn = int(np.sum((pred_b == 0) & (ref_b == 1)))
    tn = int(np.sum((pred_b == 0) & (ref_b == 0)))
    total = int(pred_b.size)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    # Diagnostic confusion map (RGB):
    # Green (34, 197, 94)   : True Positive (Confirmed Match)
    # Red (239, 68, 68)     : False Positive (Spurious / Over-prediction)
    # Blue (59, 130, 246)   : False Negative (Omission / Missed Change)
    # Dark Navy (15, 23, 42): True Negative (Background agreement)
    confusion_map = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    confusion_map[(pred_b == 0) & (ref_b == 0)] = [15, 23, 42]
    confusion_map[(pred_b == 1) & (ref_b == 1)] = [34, 197, 94]
    confusion_map[(pred_b == 1) & (ref_b == 0)] = [239, 68, 68]
    confusion_map[(pred_b == 0) & (ref_b == 1)] = [59, 130, 246]

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_pixels": total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
        "ref_binary": ref_b,
        "confusion_map": confusion_map
    }
