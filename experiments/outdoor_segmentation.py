"""Outdoor segmentation experiment — vertical slice.

Train a softmax baseline on WildScenes2d, sanity-check predictions on val.
Usage:
    pixi run python experiments/outdoor_segmentation.py --wildscenes-root /path/to/wildscenes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from experiments.candidates.softmax_baseline import SoftmaxCandidate
from experiments.datasets.wildscenes2d import WildScenes2dDataset
from segspicious import Split

# Distinct colours for the 15 WildScenes benchmark classes.
_CLASS_COLORS = np.array(
    [
        [0.00, 0.60, 0.24],  # bush
        [0.55, 0.27, 0.07],  # dirt
        [0.60, 0.60, 0.60],  # fence
        [0.47, 0.87, 0.47],  # grass
        [0.75, 0.75, 0.75],  # gravel
        [0.40, 0.26, 0.13],  # log
        [0.36, 0.25, 0.20],  # mud
        [1.00, 0.60, 0.00],  # other-object
        [0.82, 0.71, 0.55],  # other-terrain
        [0.50, 0.50, 0.50],  # rock
        [0.53, 0.81, 0.98],  # sky
        [0.86, 0.08, 0.24],  # structure
        [0.13, 0.55, 0.13],  # tree-foliage
        [0.36, 0.20, 0.09],  # tree-trunk
        [0.00, 0.45, 0.74],  # water
    ],
    dtype=np.float32,
)


def _colorize_labels(
    labels: np.ndarray,
    *,
    ignore_index: int = 255,
) -> np.ndarray:
    """Map an (H, W) integer label array to an (H, W, 3) float RGB image.

    Ignored pixels are rendered as black.
    """
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cls_id in range(len(_CLASS_COLORS)):
        mask = labels == cls_id
        rgb[mask] = _CLASS_COLORS[cls_id]
    return rgb


def evaluate_quick(
    candidate: SoftmaxCandidate,
    dataset: WildScenes2dDataset,
    *,
    n_samples: int = 5,
    output_dir: str | Path = "eval_output",
) -> None:
    """Run prediction on a few samples, print stats, and save visualisations.

    For every sample an image is saved to *output_dir* showing, side by side:
    1. The input RGB image.
    2. The predicted segmentation mask.
    3. The ground-truth segmentation mask.
    """
    ignore_index = dataset.ignore_index
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_correct = 0
    total_valid = 0

    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        output = candidate.predict(sample.image)

        valid = sample.labels != ignore_index
        n_valid = valid.sum()
        correct = (output.prediction == sample.labels) & valid
        n_correct = correct.sum()
        acc = n_correct / max(n_valid, 1)

        total_correct += n_correct
        total_valid += n_valid

        # Uncertainty stats on valid pixels.
        unc = output.predictive_uncertainty[valid]
        max_entropy = np.log(dataset.num_classes)

        print(
            f"  Sample {i:3d} — "
            f"shape: {sample.image.shape[:2]}  "
            f"acc: {acc:.3f}  "
            f"uncertainty: mean={unc.mean():.3f} "
            f"max={unc.max():.3f} "
            f"(max_possible={max_entropy:.3f})"
        )

        # --- Save side-by-side visualisation -------------------------------
        pred_rgb = _colorize_labels(output.prediction, ignore_index=ignore_index)
        gt_rgb = _colorize_labels(sample.labels, ignore_index=ignore_index)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(sample.image)
        axes[0].set_title("Input")
        axes[1].imshow(pred_rgb)
        axes[1].set_title(f"Prediction  (acc={acc:.3f})")
        axes[2].imshow(gt_rgb)
        axes[2].set_title("Ground Truth")
        for ax in axes:
            ax.axis("off")

        # Add a shared legend beneath the panels.
        cmap = ListedColormap(_CLASS_COLORS)
        handles = [
            plt.Line2D(
                [0], [0],
                marker="s",
                color="w",
                markerfacecolor=cmap(ci),
                markersize=8,
                label=name,
            )
            for ci, name in enumerate(dataset.class_names)
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(8, len(handles)),
            fontsize="x-small",
            frameon=False,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 1])

        out_path = output_dir / f"sample_{i:04d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"           → saved {out_path}")

    overall_acc = total_correct / max(total_valid, 1)
    print(f"\n  Overall pixel accuracy ({n_samples} samples): {overall_acc:.4f}")
    print(f"  Visualisations saved to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Outdoor segmentation experiment")
    parser.add_argument(
        "--wildscenes-root",
        type=str,
        required=True,
        help="Path to WildScenes root (parent of WildScenes2d/)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save/load model checkpoints",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, load from checkpoint and evaluate",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=10,
        help="Number of val samples for quick evaluation",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_output",
        help="Directory to save evaluation visualisation images",
    )
    args = parser.parse_args()

    root = args.wildscenes_root
    checkpoint_path = Path(args.checkpoint_dir) / "softmax_deeplabv3_r50.pt"

    # --- Datasets ---------------------------------------------------------
    print("Loading datasets...")
    train_data = WildScenes2dDataset(root, split=Split.TRAIN)
    val_data = WildScenes2dDataset(root, split=Split.VAL)
    print(
        f"  Train: {len(train_data)} samples, "
        f"{train_data.num_classes} classes, "
        f"ignore_index={train_data.ignore_index}"
    )
    print(f"  Val:   {len(val_data)} samples")

    # --- Candidate --------------------------------------------------------
    candidate = SoftmaxCandidate(
        name="softmax_deeplabv3_r50",
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        compile_model=not args.no_compile,
    )

    if args.eval_only:
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        candidate.load(checkpoint_path)
    else:
        # --- Train --------------------------------------------------------
        print(f"\nTraining {candidate.name}...")
        candidate.train(train_data)

        # --- Save ---------------------------------------------------------
        candidate.save(checkpoint_path)

    # --- Evaluate ---------------------------------------------------------
    print(f"\nQuick evaluation on val ({args.eval_samples} samples)...")
    evaluate_quick(
        candidate, val_data, n_samples=args.eval_samples, output_dir=args.output_dir
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
