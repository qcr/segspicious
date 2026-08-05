"""Quick sanity-check: train DeepLabV3-RN50 on a small WildScenes2d subset, evaluate on val."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.datasets.wildscenes2d import Wildscenes2dDataset
from experiments.models import DeepLabV3RN50
from segspicious import train_or_load
from segspicious.datasets import Split, subset
from segspicious.metrics import IoU, PixelAccuracy

WILDSCENES_ROOT = Path("/home/alistair/datasets/WildScenes/WildScenes2d")


def main() -> None:
    train_full = Wildscenes2dDataset(WILDSCENES_ROOT, split=Split.TRAIN)
    train_data = subset(train_full, n=100, seed=42)
    val_data = Wildscenes2dDataset(WILDSCENES_ROOT, split=Split.VAL)
    test_data = Wildscenes2dDataset(WILDSCENES_ROOT, split=Split.TEST)

    candidate = train_or_load(DeepLabV3RN50(), train_data, validation_data=val_data)
    model = candidate.model

    print(f"Model {model.name} ready.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iou = IoU(num_classes=test_data.num_classes, ignore_index=test_data.ignore_index)
    acc = PixelAccuracy(
        num_classes=test_data.num_classes, ignore_index=test_data.ignore_index
    )

    loader = DataLoader(test_data, batch_size=4, num_workers=4, pin_memory=True)

    print(f"Evaluating on val ({len(test_data)} samples) …")
    for images, labels in loader:
        images = images.to(device)
        output = model.predict(images)
        output.prediction = output.prediction.cpu()
        iou.update(output, labels)
        acc.update(output, labels)

    iou_result = iou.compute()
    acc_result = acc.compute()

    print()
    print("Per-class IoU:")
    for name, val in zip(test_data.class_names, iou_result.per_class_iou):
        print(f"  {name:20s} {val:.4f}")
    print()
    print(f"  {'mIoU':20s} {iou_result.mean_iou:.4f}")
    print(f"  {'Pixel accuracy':20s} {acc_result.pixel_accuracy:.4f}")
    print(f"  {'Mean class accuracy':20s} {acc_result.mean_class_accuracy:.4f}")


if __name__ == "__main__":
    main()
