"""DeepLabV3 ResNet-50 model with softmax UQ."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)
from tqdm import tqdm

from segspicious.datasets import SegmentationDataset
from segspicious.outputs import UncertaintyOutput

# -- ImageNet normalisation ------------------------------------------------

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _normalise(images: Tensor) -> Tensor:
    """ImageNet-normalise a ``(B, 3, H, W)`` float tensor in [0, 1]."""
    mean = _MEAN.to(images.device)
    std = _STD.to(images.device)
    return (images - mean) / std


# -- Training augmentation dataset ----------------------------------------


class _TrainAugDataset(Dataset):
    """Wraps a SegmentationDataset with joint image/label augmentation."""

    def __init__(
        self,
        dataset: SegmentationDataset,
        crop_size: int,
        scale_range: tuple[float, float] = (0.5, 2.0),
        ignore_index: int = 255,
    ) -> None:
        self._dataset = dataset
        self._crop_size = crop_size
        self._scale_lo, self._scale_hi = scale_range
        self._ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, labels = self._dataset[index]
        # image: (3, H, W) float [0, 1], labels: (H, W) long

        _, h, w = image.shape

        # -- Random scale --------------------------------------------------
        scale = random.uniform(self._scale_lo, self._scale_hi)
        new_h, new_w = int(h * scale), int(w * scale)

        image = F.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        labels = (
            F.interpolate(
                labels.float().unsqueeze(0).unsqueeze(0),
                size=(new_h, new_w),
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
            .long()
        )

        # -- Pad if smaller than crop_size ---------------------------------
        pad_h = max(self._crop_size - new_h, 0)
        pad_w = max(self._crop_size - new_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
            labels = F.pad(labels, (0, pad_w, 0, pad_h), value=self._ignore_index)

        # -- Random crop ---------------------------------------------------
        _, ch, cw = image.shape
        top = random.randint(0, ch - self._crop_size)
        left = random.randint(0, cw - self._crop_size)
        image = image[:, top : top + self._crop_size, left : left + self._crop_size]
        labels = labels[top : top + self._crop_size, left : left + self._crop_size]

        # -- Random horizontal flip ----------------------------------------
        if random.random() > 0.5:
            image = image.flip(-1)
            labels = labels.flip(-1)

        # -- ImageNet normalise --------------------------------------------
        image = _normalise(image)

        return image, labels


# -- Model -----------------------------------------------------------------


class DeepLabV3RN50:
    """DeepLabV3 ResNet-50 segmentation model with softmax UQ.

    Uses a COCO-pretrained DeepLabV3 ResNet-50 backbone from torchvision,
    fine-tuned on the provided dataset.  Produces softmax class
    probabilities and predictive entropy as uncertainty.

    The class *is* the configuration — no constructor arguments.
    Dataset-dependent values like ``num_classes`` are discovered from
    the dataset at train time.  Different hyperparameters = different
    class (use inheritance to override class attributes).
    """

    epochs: int = 50
    batch_size: int = 4
    lr: float = 0.01
    crop_size: int = 512
    num_workers: int = 4
    log_dir: str = "runs"

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._num_classes: int | None = None

    @staticmethod
    def _build_model(num_classes: int) -> nn.Module:
        model = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
        )

        # Replace classifier heads for target class count
        in_ch = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_ch, num_classes, 1)

        in_ch_aux = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, num_classes, 1)

        return model

    @property
    def name(self) -> str:
        return "deeplabv3-rn50"

    # -- Training ----------------------------------------------------------

    def train(
        self,
        dataset: SegmentationDataset,
        validation_data: SegmentationDataset | None = None,
    ) -> None:
        """Fine-tune on the given dataset.

        Discovers ``num_classes`` from the dataset, builds the model,
        trains for ``self.epochs`` epochs, and stores the best weights.

        When *validation_data* is provided the model tracks validation
        loss each epoch and restores the best-performing weights before
        returning (best-validation checkpointing).

        Training and validation metrics are logged to TensorBoard under
        ``{self.log_dir}/{self.name}``.

        Args:
            dataset: Training data.
            validation_data: Optional validation split for monitoring
                training progress and best-checkpoint selection.
        """
        self._num_classes = dataset.num_classes
        self._model = self._build_model(self._num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = self._model.to(device)
        model.train()

        train_ds = _TrainAugDataset(
            dataset,
            crop_size=self.crop_size,
            ignore_index=dataset.ignore_index,
        )
        loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )

        # Backbone at lr, fresh heads at 10× lr
        head_params: list[nn.Parameter] = []
        backbone_params: list[nn.Parameter] = []
        for name, param in model.named_parameters():
            if "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimiser = SGD(
            [
                {"params": backbone_params, "lr": self.lr},
                {"params": head_params, "lr": self.lr * 10},
            ],
            momentum=0.9,
            weight_decay=1e-4,
        )

        max_iters = self.epochs * len(loader)
        scheduler = LambdaLR(
            optimiser,
            lr_lambda=lambda it: (1 - it / max_iters) ** 0.9,
        )

        criterion = nn.CrossEntropyLoss(ignore_index=dataset.ignore_index)

        # Optional validation loader
        val_loader: DataLoader | None = None
        if validation_data is not None:
            val_loader = DataLoader(
                validation_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=device.type == "cuda",
            )

        # -- TensorBoard writer -------------------------------------------
        writer = SummaryWriter(log_dir=str(Path(self.log_dir) / f"{self.name}-{dataset.name}"))

        # -- Best-validation checkpoint state -----------------------------
        best_val_loss = float("inf")
        best_state_dict: dict | None = None

        global_step = 0
        for epoch in range(self.epochs):
            model.train()
            epoch_loss_sum = 0.0
            epoch_steps = 0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)

                out = model(images)
                loss = criterion(out["out"], labels)
                loss_aux = criterion(out["aux"], labels)
                total_loss = loss + 0.4 * loss_aux

                optimiser.zero_grad()
                total_loss.backward()
                optimiser.step()
                scheduler.step()

                step_loss = total_loss.item()
                epoch_loss_sum += step_loss
                epoch_steps += 1
                global_step += 1

                writer.add_scalar("train/loss_step", step_loss, global_step)
                pbar.set_postfix(loss=f"{step_loss:.4f}")

            # Log epoch-level training loss
            epoch_train_loss = epoch_loss_sum / max(epoch_steps, 1)
            writer.add_scalar("train/loss_epoch", epoch_train_loss, epoch + 1)

            # Log learning rate
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch + 1)

            # -- Validation loss ------------------------------------------
            if val_loader is not None:
                model.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = _normalise(images.to(device))
                        labels = labels.to(device)
                        out = model(images)
                        val_loss_sum += criterion(out["out"], labels).item() * labels.size(0)
                        val_count += labels.size(0)
                val_loss = val_loss_sum / max(val_count, 1)
                writer.add_scalar("val/loss", val_loss, epoch + 1)
                print(f"  val_loss={val_loss:.4f}")

                # Track best validation checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = copy.deepcopy(model.state_dict())

        writer.close()

        # -- Restore best weights -----------------------------------------
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            print(f"  Restored best val checkpoint (val_loss={best_val_loss:.4f})")

        self._model = model.cpu()

    # -- Inference ---------------------------------------------------------

    def predict(self, images: Tensor) -> UncertaintyOutput:
        """Single forward pass → softmax probabilities + entropy."""
        assert self._model is not None, (
            "Model not initialised. Call train() or load() first."
        )
        self._model.to(images.device)
        self._model.eval()

        with torch.no_grad():
            logits = self._model(_normalise(images))["out"]  # (B, C, H, W)

            class_probs = F.softmax(logits, dim=1)  # (B, C, H, W)
            prediction = class_probs.argmax(dim=1)  # (B, H, W)

            # Predictive entropy: -Σ p·log(p)
            log_probs = torch.log(class_probs.clamp(min=1e-8))
            predictive_uncertainty = -(class_probs * log_probs).sum(dim=1)  # (B, H, W)

        return UncertaintyOutput(
            prediction=prediction,
            class_probs=class_probs,
            predictive_uncertainty=predictive_uncertainty,
        )

    # -- Serialisation -----------------------------------------------------

    def save(self, path: Path) -> None:
        """Save model weights and num_classes to disk."""
        assert self._model is not None, (
            "Model not initialised. Call train() before save()."
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self._model.state_dict(), "num_classes": self._num_classes},
            path,
        )

    def load(self, path: Path) -> None:
        """Load model weights and num_classes from disk."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self._num_classes = checkpoint["num_classes"]
        self._model = self._build_model(self._num_classes)
        self._model.load_state_dict(checkpoint["state_dict"])
