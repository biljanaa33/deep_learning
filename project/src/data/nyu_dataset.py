# load RGB/depth correctly.
# src/data/nyu_dataset.py

from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

MAX_DEPTH_M = 10.0


def load_depth(depth_path):
    depth_img = Image.open(depth_path)
    depth_raw = np.array(depth_img)

    if depth_raw.dtype == np.uint16:
        depth = depth_raw.astype(np.float32) / 1000.0
    elif depth_raw.dtype == np.uint8:
        depth = depth_raw.astype(np.float32) / 255.0 * MAX_DEPTH_M
    else:
        raise ValueError(f"Unsupported depth dtype: {depth_raw.dtype}")

    return depth


def read_existing_pairs(csv_path, root):
    pairs = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)

        for rgb_rel, depth_rel in reader:
            rgb_path = root / rgb_rel
            depth_path = root / depth_rel

            if rgb_path.exists() and depth_path.exists():
                pairs.append((rgb_path, depth_path))

    return pairs


class NYUDepthDataset(Dataset):
    def __init__(self, pairs, image_size=(240, 320)):
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.pairs[idx]
        height, width = self.image_size

        rgb = Image.open(rgb_path).convert("RGB")
        depth = load_depth(depth_path)

        rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.resize((width, height), Image.Resampling.NEAREST)

        rgb = np.array(rgb).astype(np.float32) / 255.0
        depth = np.array(depth_img).astype(np.float32)

        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        depth = torch.from_numpy(depth).unsqueeze(0)

        mask = (depth > 0.0) & (depth <= MAX_DEPTH_M)

        return {
            "image": rgb,
            "depth": depth,
            "mask": mask.float(),
        }
