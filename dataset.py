"""Dataset for knee X-ray bone (tibia) segmentation."""
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def resolve(path, root):
    """Use `path` as-is if it exists, otherwise look for its file name inside `root`."""
    if os.path.exists(path):
        return path
    return os.path.join(root, os.path.basename(path))


class KneeDataset(Dataset):
    def __init__(self, csv_path, image_root, mask_root=None, img_size=224, augment=False):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.mask_root = mask_root if mask_root is not None else image_root
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load(self, path, is_mask):
        img = Image.open(path).convert("L")
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        img = img.resize((self.img_size, self.img_size), resample)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if is_mask:
            arr = (arr > 0.5).astype(np.float32)
        return arr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = resolve(str(row["xrays"]), self.image_root)
        image = self._load(img_path, is_mask=False)

        has_mask = str(row["masks"]) not in ("None", "nan", "")
        if has_mask:
            mask = self._load(resolve(str(row["masks"]), self.mask_root), is_mask=True)
        else:
            mask = np.zeros_like(image)

        if self.augment:
            if np.random.rand() < 0.5:                       # horizontal flip
                image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
            image = np.clip(image * np.random.uniform(0.9, 1.1), 0, 1)  # brightness

        return (torch.from_numpy(image).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0),
                os.path.basename(img_path),
                has_mask)
