#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch Extraction Pipeline with Hardcoded Paths (PyTorch)
========================================================
- Loads real images (GeoTIFF) as torch.Tensor
- Extracts 32×32 patches from two input images
- Produces arrays of patches (flattened columns)
- Produces ground truth patch medians
- Saves output to `.mat`
"""

import torch
import numpy as np
import logging
import scipy.io as sio
import rasterio

from torchvision import transforms
from PIL import Image

# ======================
# Logging Setup
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ======================
# Image Loader
# ======================
def load_geotiff(path: str) -> torch.Tensor:
    with rasterio.open(path) as src:
        img = src.read()  # (C,H,W)
    logger.info(f"Loaded GeoTIFF: {path} with shape {img.shape}")
    return torch.from_numpy(img).float()


# ======================
# Patch Extraction Functions
# ======================
def extract_patches(image: torch.Tensor, patch_size=32, stride=32) -> torch.Tensor:
    logger.info(f"Extracting patches from image with shape: {image.shape}")
    C, H, W = image.shape
    patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, C, patch_size, patch_size)
    patches_flat = patches.view(patches.size(0), -1).T
    logger.info(f"Extracted {patches.size(0)} patches each of size {patch_size}×{patch_size}")
    return patches_flat


def extract_gt_patches(gt_image: torch.Tensor, patch_size=32, stride=32) -> torch.Tensor:
    logger.info(f"Extracting ground truth patches from GT shape: {gt_image.shape}")
    patches = gt_image.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
    patches = patches.contiguous().view(-1, patch_size * patch_size)
    medians = patches.median(dim=1).values
    logger.info(f"Computed medians for {medians.shape[0]} ground truth patches.")
    return medians


# ======================
# Main pipeline with hardcoded paths
# ======================
def process_and_save():
    logger.info("🚀 Starting patch extraction pipeline...")

    data_path ="data"
    # 🔷 Hardcoded paths
    pre_image_path = f"{data_path}/T35TMF_20230729T090559_B01.tif"
    post_image_path = f"{data_path}/T35TMF_20230912T090601_B01.tif"
    gt_image_path = f"{data_path}/gt.tif"
    output_mat_path = f"{data_path}/output/Patches_Array.mat"

    patch_size = 32
    stride = 32

    # 🔷 Load images
    image1 = load_geotiff(pre_image_path)
    image2 = load_geotiff(post_image_path)
    gt = load_geotiff(gt_image_path).squeeze(0)  # assume single band

    # 🔷 Process
    img1_patches = extract_patches(image1, patch_size, stride)
    img2_patches = extract_patches(image2, patch_size, stride)
    gt_medians = extract_gt_patches(gt, patch_size, stride)

    # 🔷 Save as .mat
    sio.savemat(output_mat_path, {
        "img1_patches": img1_patches.numpy(),
        "img2_patches": img2_patches.numpy(),
        "gt_medians": gt_medians.numpy()
    })
    logger.info(f"✅ Saved patch data to {output_mat_path}")


if __name__ == "__main__":
    process_and_save()
