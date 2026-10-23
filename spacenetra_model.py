"""
spacenetra_model.py
===================
Experimental Deep Learning Change Detection Model for Space Netra.
Architecture: 6-channel Bitemporal U-Net trained on S2Looking subset in Google Colab.
Input: Two RGB images (Earlier + Later) resized to 256x256, concatenated along channel dimension.
Output: Single-channel binary change probability map.
"""

import os
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

DEFAULT_CHECKPOINT_PATH = os.path.join("models", "spacenetra_s2looking_unet.pth")


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

        metadata = {
            "epoch": checkpoint.get("epoch", 2),
            "image_size": checkpoint.get("image_size", 256),
            "input_channels": checkpoint.get("input_channels", 6),
            "output_channels": checkpoint.get("output_channels", 1),
            "validation_metrics": checkpoint.get("validation_metrics", {
                "precision": 0.2993,
                "recall": 0.1534,
                "f1": 0.2028,
                "iou": 0.1129,
                "loss": 1.1263
            }),
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

    changed_count = int(np.sum(binary_mask))
    total_pixels = int(binary_mask.size)
    change_ratio_pct = round((changed_count / max(1, total_pixels)) * 100.0, 2)

    return {
        "prob_map": prob_map,
        "binary_mask": binary_mask,
        "overlay_img": overlay,
        "changed_pixels": changed_count,
        "total_pixels": total_pixels,
        "change_ratio_pct": change_ratio_pct,
        "threshold": threshold,
        "is_valid": bool(np.isfinite(prob_map).all())
    }
