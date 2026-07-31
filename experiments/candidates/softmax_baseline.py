"""Softmax baseline candidate — single-network segmentation with entropy UQ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

from segspicious import (
    SegmentationDataset,
    TorchDatasetAdapter,
    UncertaintyOutput,
)

# Pre-allocated normalisation constants (avoids per-sample tensor creation).
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _train_transform(
    image: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random crop + horizontal flip for training augmentation.

    Operates on (C, H, W) image tensor and (H, W) label tensor.
    """
    # Random resize: scale ∈ [0.5, 2.0], then crop to 512×512.
    scale = float(torch.empty(1).uniform_(0.5, 2.0).item())
    _, h, w = image.shape
    new_h, new_w = int(h * scale), int(w * scale)

    image = F.interpolate(
        image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    labels = (
        F.interpolate(
            labels.float().unsqueeze(0).unsqueeze(0),
            size=(new_h, new_w),
            mode="nearest",
        )
        .long()
        .squeeze(0)
        .squeeze(0)
    )

    # Pad if smaller than crop size.
    crop_h, crop_w = 512, 512
    pad_h = max(crop_h - new_h, 0)
    pad_w = max(crop_w - new_w, 0)
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
        labels = F.pad(labels, (0, pad_w, 0, pad_h), value=255)

    # Random crop.
    _, ph, pw = image.shape
    top = torch.randint(0, ph - crop_h + 1, (1,)).item()
    left = torch.randint(0, pw - crop_w + 1, (1,)).item()
    image = image[:, top : top + crop_h, left : left + crop_w]
    labels = labels[top : top + crop_h, left : left + crop_w]

    # Random horizontal flip.
    if torch.rand(1).item() > 0.5:
        image = image.flip(-1)
        labels = labels.flip(-1)

    # ImageNet normalisation.
    image = (image - _IMAGENET_MEAN) / _IMAGENET_STD

    return image, labels


def _eval_transform(
    image: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalise only — no augmentation."""
    image = (image - _IMAGENET_MEAN) / _IMAGENET_STD
    return image, labels


class SoftmaxCandidate:
    """Single-network DeepLabV3-ResNet50 with softmax confidence.

    Produces :class:`UncertaintyOutput` with ``prediction``,
    ``class_probs`` (softmax), and ``predictive_uncertainty``
    (entropy of the softmax distribution).

    Parameters
    ----------
    name:
        Identifier for results tables and saved state.
    epochs:
        Number of training epochs.
    lr:
        Initial learning rate for SGD.
    batch_size:
        Training batch size.
    num_workers:
        DataLoader worker count.
    pretrained_backbone:
        Whether to use ImageNet-pretrained ResNet50 weights.
    use_amp:
        Enable automatic mixed precision (FP16) training.  Greatly
        reduces memory and increases throughput on modern GPUs.
    compile_model:
        Apply ``torch.compile`` to the model for additional speed.
    """

    def __init__(
        self,
        name: str = "softmax_deeplabv3_r50",
        *,
        epochs: int = 50,
        lr: float = 0.01,
        batch_size: int = 24,
        num_workers: int = 4,
        pretrained_backbone: bool = True,
        use_amp: bool = True,
        compile_model: bool = True,
    ) -> None:
        self._name = name
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pretrained_backbone = pretrained_backbone
        self._use_amp = use_amp
        self._compile_model = compile_model

        self._model: nn.Module | None = None
        self._num_classes: int | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Candidate protocol ------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    def train(self, dataset: SegmentationDataset) -> None:
        """Train the model on the given dataset."""
        self._num_classes = dataset.num_classes
        ignore_index = dataset.ignore_index

        # Enable cuDNN auto-tuner — input size is fixed at 512×512 after
        # the random-crop augmentation, so this is a pure win.
        torch.backends.cudnn.benchmark = True

        # Build model.
        self._model = self._build_model(self._num_classes)
        self._model.to(self._device)

        if self._compile_model and hasattr(torch, "compile"):
            self._model = torch.compile(self._model)

        # Data loading.
        torch_dataset = TorchDatasetAdapter(dataset, transform=_train_transform)
        use_persistent = self._num_workers > 0
        loader = DataLoader(
            torch_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=use_persistent,
            prefetch_factor=4 if use_persistent else None,
        )

        # Optimiser & schedule.
        optimiser = torch.optim.SGD(
            self._model.parameters(),
            lr=self._lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimiser,
            lr_lambda=lambda epoch: (1 - epoch / self._epochs) ** 0.9,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # AMP scaler — only meaningful on CUDA.
        use_amp = self._use_amp and self._device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)

        # Training loop.
        self._model.train()
        for epoch in range(1, self._epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(
                loader,
                desc=f"  Epoch {epoch:3d}/{self._epochs}",
                leave=False,
            )
            for images, labels in pbar:
                images = images.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=self._device.type, enabled=use_amp
                ):
                    logits = self._model(images)["out"]  # (B, C, H, W)

                    # DeepLabV3 output may differ in size from labels if
                    # input wasn't divisible by output stride — resize
                    # logits to match.
                    if logits.shape[2:] != labels.shape[1:]:
                        logits = F.interpolate(
                            logits,
                            size=labels.shape[1:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    loss = criterion(logits, labels)

                optimiser.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()

                epoch_loss += loss.item()
                n_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            lr_now = optimiser.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}/{self._epochs} — "
                f"loss: {avg_loss:.4f}  lr: {lr_now:.6f}"
            )

    def predict(self, image: np.ndarray) -> UncertaintyOutput:
        """Run inference on a single (H, W, C) uint8 RGB image."""
        if self._model is None:
            raise RuntimeError("Model not initialised — call train() or load() first.")

        self._model.eval()

        # Preprocess: same normalisation as eval_transform.
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(self._device)  # (1, C, H, W)

        with torch.no_grad():
            logits = self._model(tensor)["out"]  # (1, C, H, W)

            # Resize to input spatial dims if needed.
            h, w = image.shape[:2]
            if logits.shape[2:] != (h, w):
                logits = F.interpolate(
                    logits, size=(h, w), mode="bilinear", align_corners=False
                )

            probs = F.softmax(logits, dim=1)  # (1, C, H, W)

        # Move to CPU / numpy.
        probs_np = probs.squeeze(0).cpu().numpy()  # (C, H, W)
        probs_np = probs_np.transpose(1, 2, 0)  # (H, W, C)

        prediction = probs_np.argmax(axis=2)  # (H, W)

        # Predictive uncertainty = entropy of softmax distribution.
        # H = -Σ p log p,  with 0 log 0 = 0.
        entropy = -np.sum(
            probs_np * np.log(np.clip(probs_np, 1e-10, 1.0)), axis=2
        )  # (H, W)

        return UncertaintyOutput(
            prediction=prediction,
            class_probs=probs_np,
            predictive_uncertainty=entropy,
        )

    def save(self, path: Path) -> None:
        """Save model weights to *path*."""
        if self._model is None:
            raise RuntimeError("No model to save — call train() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "num_classes": self._num_classes,
            },
            path,
        )
        print(f"  Saved checkpoint to {path}")

    def load(self, path: Path) -> None:
        """Load model weights from *path*."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        self._num_classes = checkpoint["num_classes"]
        self._model = self._build_model(self._num_classes)

        # torch.compile wraps the model and prefixes state-dict keys with
        # "_orig_mod.".  Strip the prefix so we can load into a vanilla model.
        state_dict = checkpoint["model_state_dict"]
        prefix = "_orig_mod."
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}

        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()
        print(f"  Loaded checkpoint from {path}")

    # -- Internals ---------------------------------------------------------

    def _build_model(self, num_classes: int) -> nn.Module:
        """Construct a DeepLabV3-ResNet50."""
        model = deeplabv3_resnet50(
            weights=None,
            weights_backbone="IMAGENET1K_V1" if self._pretrained_backbone else None,
            num_classes=num_classes,
        )
        return model
