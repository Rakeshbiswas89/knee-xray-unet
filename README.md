# Tibia Segmentation in Knee X-rays with U-Net

Course project: bone segmentation on knee X-ray images with a U-Net (PyTorch).
**Group 4, structure: tibia.**

Author: Rakesh Biswas (sa21761), SAMK

## Task
Segment the tibia in knee X-rays. The group dataset has 150 images:
100 manually annotated (binary masks, made in Photoshop) and 50 unlabeled test images.
The 100 annotated images are split into **80 train / 20 validation**.

## Project structure
```
.
├── data/                 # CSV files (xrays, masks): dataset, train, val, test
├── dataset.py            # KneeDataset: loading, resizing, binarizing masks, light augmentation
├── model.py              # U-Net (4 down / 4 up levels, base 32 channels)
├── utils.py              # soft Dice loss and hard Dice score
├── train.py              # training loop, saves best model by validation Dice
├── evaluate.py           # Dice score + predicted masks and overlays
├── requirements.txt
└── checkpoints/          # best_model.pth is written here (git-ignored)
```

## Approach
- **Preprocessing:** grayscale, kept at 224x224 (the native size), scaled to [0, 1]. Masks are resized with nearest neighbour and binarized.
- **Augmentation (train only):** random horizontal flip and small brightness change.
- **Model:** U-Net with skip connections, BatchNorm, 1 output logit channel.
- **Loss:** BCE with logits + soft Dice loss (`1 - Dice` on sigmoid probabilities).
- **Metric:** hard Dice (probabilities thresholded at 0.5), as in `utils.py`.
- **Model selection:** the epoch with the best validation Dice is saved to `checkpoints/best_model.pth`.

## How to run
```bash
pip install -r requirements.txt

# train (set --image_root to the folder with your Group 4 images, --mask_root to your masks folder)
python train.py --image_root "/path/to/Group 4" --mask_root "/path/to/masks" --epochs 50

# evaluate the saved best model on the validation set (labeled, gives Dice)
python evaluate.py --csv data/val.csv --image_root "/path/to/Group 4" --mask_root "/path/to/masks"

# predict masks for the 50 unlabeled test images
python evaluate.py --csv data/test.csv --image_root "/path/to/Group 4"
```
Predicted masks and red overlays are written to `outputs/predictions/`.

## Results
| Split | Images | Dice |
|---|---|---|
| Validation | 20 | _fill in after training_ |

The 50 test images have no ground-truth masks in `data/test.csv`, so predicted masks are produced for them
and the reported Dice is computed on the labeled validation split.

## Data note
The dataset images are course material and are not included in this repository.
The CSV files store file names only; `--image_root` / `--mask_root` tell the code where the files are.
