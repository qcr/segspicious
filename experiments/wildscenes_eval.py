"""Quick sanity-check: train DeepLabV3-RN50 on a small WildScenes2d subset, evaluate on val."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.candidates import DeepLabV3RN50Candidate
from experiments.datasets.wildscenes2d import Wildscenes2dDataset
from segspicious.datasets import Split, subset
from segspicious.metrics import IoU, PixelAccuracy

WILDSCENES_ROOT = Path("/home/alistair/datasets/WildScenes/WildScenes2d")
CHECKPOINT_DIR = Path("checkpoints")


def main() -> None:
    train_full = Wildscenes2dDataset(WILDSCENES_ROOT, split=Split.TRAIN)
    train_data = subset(train_full, n=100, seed=42)
    val_data = Wildscenes2dDataset(WILDSCENES_ROOT, split=Split.VAL)

    candidate = DeepLabV3RN50Candidate(
        num_classes=train_data.num_classes,
        epochs=5,
        batch_size=4,
        lr=0.01,
        crop_size=512,
        num_workers=4,
    )

    print(f"Training {candidate.name} …")
    candidate.train(train_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iou = IoU(num_classes=val_data.num_classes, ignore_index=val_data.ignore_index)
    acc = PixelAccuracy(
        num_classes=val_data.num_classes, ignore_index=val_data.ignore_index
    )

    loader = DataLoader(val_data, batch_size=4, num_workers=4, pin_memory=True)

    print(f"Evaluating on val ({len(val_data)} samples) …")
    for images, labels in loader:
        images = images.to(device)
        output = candidate.predict(images)
        output.prediction = output.prediction.cpu()
        iou.update(output, labels)
        acc.update(output, labels)

    iou_result = iou.compute()
    acc_result = acc.compute()

    print()
    print("Per-class IoU:")
    for name, val in zip(val_data.class_names, iou_result.per_class_iou):
        print(f"  {name:20s} {val:.4f}")
    print()
    print(f"  {'mIoU':20s} {iou_result.mean_iou:.4f}")
    print(f"  {'Pixel accuracy':20s} {acc_result.pixel_accuracy:.4f}")
    print(f"  {'Mean class accuracy':20s} {acc_result.mean_class_accuracy:.4f}")


if __name__ == "__main__":
    main()
