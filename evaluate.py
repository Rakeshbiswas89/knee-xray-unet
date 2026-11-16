"""Evaluate the best saved model, report Dice, and save predicted masks + overlays."""
import argparse
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import KneeDataset
from model import UNet
from train import get_device
from utils import dice_score_from_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/test.csv")
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--mask_root", default=None)
    ap.add_argument("--weights", default="checkpoints/best_model.pth")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--out_dir", default="outputs/predictions")
    args = ap.parse_args()

    device = get_device()
    ds = KneeDataset(args.csv, args.image_root, args.mask_root, args.img_size)
    dl = DataLoader(ds, batch_size=4)

    model = UNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    total, n_labeled = 0.0, 0
    with torch.no_grad():
        for x, y, names, has_mask in dl:
            logits = model(x.to(device))
            preds = (torch.sigmoid(logits) > 0.5).float().cpu()
            for i, name in enumerate(names):
                img = (x[i, 0].numpy() * 255).astype(np.uint8)
                pred = (preds[i, 0].numpy() * 255).astype(np.uint8)
                overlay = np.stack([img, img, img], axis=-1)
                overlay[pred > 0] = (0.55 * overlay[pred > 0] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
                Image.fromarray(pred).save(os.path.join(args.out_dir, f"mask_{name}"))
                Image.fromarray(overlay).save(os.path.join(args.out_dir, f"overlay_{name}"))
            if bool(has_mask.all()):
                total += dice_score_from_logits(logits.cpu(), y) * x.size(0)
                n_labeled += x.size(0)

    print(f"Predicted masks saved to {args.out_dir}/ ({len(ds)} images)")
    if n_labeled:
        print(f"Dice score on {n_labeled} labeled images: {total / n_labeled:.4f}")
    else:
        print("No ground-truth masks in this CSV, so no Dice score can be computed. "
              "Evaluate on data/val.csv to report Dice.")


if __name__ == "__main__":
    main()
