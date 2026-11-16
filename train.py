"""Train a U-Net on the manually annotated knee X-rays. Saves the best model by validation Dice."""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import KneeDataset
from model import UNet
from utils import dice_loss_from_logits, dice_score_from_logits


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--val_csv", default="data/val.csv")
    ap.add_argument("--image_root", required=True, help="folder with the X-ray images")
    ap.add_argument("--mask_root", default=None, help="folder with the masks (defaults to image_root)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--save_path", default="checkpoints/best_model.pth")
    args = ap.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Device: {device}")

    train_ds = KneeDataset(args.train_csv, args.image_root, args.mask_root, args.img_size, augment=True)
    val_ds = KneeDataset(args.val_csv, args.image_root, args.mask_root, args.img_size)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    model = UNet().to(device)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y, _, _ in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = bce(logits, y) + dice_loss_from_logits(logits, y)   # BCE + soft Dice
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_ds)

        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for x, y, _, _ in val_dl:
                x, y = x.to(device), y.to(device)
                val_dice += dice_score_from_logits(model(x), y) * x.size(0)   # hard Dice
        val_dice /= len(val_ds)

        flag = ""
        if val_dice > best:
            best = val_dice
            torch.save(model.state_dict(), args.save_path)
            flag = "  <- saved best"
        print(f"Epoch {epoch:3d}/{args.epochs} | train loss {tr_loss:.4f} | val Dice {val_dice:.4f}{flag}")

    print(f"Best validation Dice: {best:.4f} (saved to {args.save_path})")


if __name__ == "__main__":
    main()
