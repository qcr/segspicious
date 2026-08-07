"""PASTEL DINOv2 model: frozen DINOv2 backbone + MLP head.

Reference: PASTEL: A Good Foundation is Worth Many Labels
(Vödisch et al., RA-L 2025, https://arxiv.org/abs/2405.19035)
"""

from __future__ import annotations

import copy
import random
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from segspicious.datasets import SegmentationDataset, balanced_subset
from segspicious.outputs import UncertaintyOutput

# -- ImageNet normalisation ------------------------------------------------

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _normalise(images: Tensor) -> Tensor:
    """ImageNet-normalise a ``(B, 3, H, W)`` float tensor in [0, 1]."""
    mean = _MEAN.to(images.device)
    std = _STD.to(images.device)
    return (images - mean) / std


# -- PASTEL segmentation module --------------------------------------------


class _PastelSegmenter(nn.Module):
    """Frozen DINOv2 backbone + 4-layer MLP head (1×1 convs).

    Only the MLP head (~600K params) is trainable.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "dinov2_vitl14",
        feat_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.patch_size = 14

        # Frozen DINOv2 backbone
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is")
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.encoder = torch.hub.load(
                "facebookresearch/dinov2", backbone, pretrained=True
            )
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # MLP head (1×1 convolutions) — only trainable part
        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, 300, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(300, 300, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(300, 200, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(200, num_classes, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: ``(B, 3, H, W)`` input, H and W divisible by 14.

        Returns:
            logits: ``(B, C, H, W)``
        """
        B, _, H, W = x.shape
        ph, pw = H // self.patch_size, W // self.patch_size

        with torch.no_grad():
            features = self.encoder.forward_features(x)
            tokens = features["x_norm_patchtokens"]  # (B, ph*pw, feat_dim)

        # Reshape to spatial grid and upsample to input resolution
        feat = tokens.permute(0, 2, 1).reshape(B, -1, ph, pw)
        feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        return self.head(feat)

    def train(self, mode: bool = True):
        """Override to keep encoder always in eval mode."""
        super().train(mode)
        self.encoder.eval()
        return self


# -- Training augmentation dataset ----------------------------------------


class _TrainAugDataset(Dataset):
    """Wraps a SegmentationDataset with PASTEL-specific augmentations.

    Joint image/label transforms:
    - RandomResizedCrop to crop_size (scale 0.2–1.0, ratio 0.75–1.333)
    - Random horizontal flip

    Image-only transforms:
    - Colour jitter (brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    - ImageNet normalisation
    """

    def __init__(
        self,
        dataset: SegmentationDataset,
        crop_size: int,
        ignore_index: int = 255,
    ) -> None:
        self._dataset = dataset
        self._crop_size = crop_size
        self._ignore_index = ignore_index
        self._colour_jitter = T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, labels = self._dataset[index]
        # image: (3, H, W) float [0, 1], labels: (H, W) long

        _, h, w = image.shape

        # -- RandomResizedCrop (joint) ------------------------------------
        # Compute crop parameters
        scale = random.uniform(0.2, 1.0)
        ratio = random.uniform(0.75, 1.333)
        area = h * w * scale
        crop_h = int(round((area / ratio) ** 0.5))
        crop_w = int(round((area * ratio) ** 0.5))
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        image = image[:, top : top + crop_h, left : left + crop_w]
        labels = labels[top : top + crop_h, left : left + crop_w]

        # Resize to crop_size
        image = F.interpolate(
            image.unsqueeze(0),
            size=(self._crop_size, self._crop_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        labels = (
            F.interpolate(
                labels.float().unsqueeze(0).unsqueeze(0),
                size=(self._crop_size, self._crop_size),
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
            .long()
        )

        # -- Random horizontal flip ---------------------------------------
        if random.random() > 0.5:
            image = image.flip(-1)
            labels = labels.flip(-1)

        # -- Colour jitter (image only) -----------------------------------
        image = self._colour_jitter(image)

        # -- ImageNet normalise -------------------------------------------
        image = _normalise(image)

        return image, labels


# -- Model -----------------------------------------------------------------


class PastelDINOv2:
    """PASTEL DINOv2 ViT-L/14 segmentation model with softmax UQ.

    Frozen DINOv2 backbone with a tiny trainable MLP head (~600K params).
    Designed for sample-efficient training with hard pixel mining and
    class-balanced subset selection.

    The class *is* the configuration — no constructor arguments.
    """

    backbone: str = "dinov2_vitl14"
    feat_dim: int = 1024
    epochs: int = 150
    batch_size: int = 4
    lr: float = 1e-3
    crop_size: int = 504
    hard_mining_ratio: float = 0.2
    num_train_samples: int | None = None
    num_workers: int = 4
    val_interval: int = 10
    log_dir: str = "runs"

    def __init__(self) -> None:
        self._model: _PastelSegmenter | None = None
        self._num_classes: int | None = None

    def _build_model(self, num_classes: int) -> _PastelSegmenter:
        return _PastelSegmenter(
            num_classes=num_classes,
            backbone=self.backbone,
            feat_dim=self.feat_dim,
        )

    @property
    def name(self) -> str:
        base = f"pastel-{self.backbone.replace('_', '-')}"
        if self.num_train_samples is not None:
            return f"{base}-n{self.num_train_samples}"
        return base

    # -- Training ----------------------------------------------------------

    def train(
        self,
        dataset: SegmentationDataset,
        validation_data: SegmentationDataset | None = None,
    ) -> None:
        """Train the MLP head on the given dataset.

        If ``num_train_samples`` is set and less than ``len(dataset)``,
        applies :func:`balanced_subset` internally to select a
        class-balanced training subset.

        Uses hard pixel mining (top-k% hardest pixels), Adam with
        cosine annealing, and PASTEL-specific augmentations.
        """
        self._num_classes = dataset.num_classes
        ignore_index = dataset.ignore_index

        # Apply balanced subset if configured
        train_data = dataset
        if self.num_train_samples is not None and self.num_train_samples < len(dataset):
            train_data = balanced_subset(dataset, n=self.num_train_samples)

        self._model = self._build_model(self._num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._model.to(device)
        model.train()

        train_ds = _TrainAugDataset(
            train_data,
            crop_size=self.crop_size,
            ignore_index=ignore_index,
        )
        loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )

        # Only head parameters are trainable
        optimiser = Adam(model.head.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimiser, T_max=self.epochs)

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

                logits = model(images)  # (B, C, H, W)

                # Hard pixel mining: compute per-pixel loss, keep top-k%
                loss = F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=ignore_index,
                    reduction="none",
                )  # (B, H, W)
                k = max(1, int(self.hard_mining_ratio * loss.numel()))
                loss = torch.topk(loss.reshape(-1), k).values.mean()

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                step_loss = loss.item()
                epoch_loss_sum += step_loss
                epoch_steps += 1
                global_step += 1

                writer.add_scalar("train/loss_step", step_loss, global_step)
                pbar.set_postfix(loss=f"{step_loss:.4f}")

            scheduler.step()

            # Log epoch-level training loss
            epoch_train_loss = epoch_loss_sum / max(epoch_steps, 1)
            writer.add_scalar("train/loss_epoch", epoch_train_loss, epoch + 1)

            # Log learning rate
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch + 1)

            # -- Validation -----------------------------------------------
            is_last_epoch = epoch == self.epochs - 1
            if val_loader is not None and (is_last_epoch or (epoch + 1) % self.val_interval == 0):
                model.eval()
                val_loss_sum = 0.0
                val_count = 0
                total_intersection = torch.zeros(self._num_classes, device=device)
                total_union = torch.zeros(self._num_classes, device=device)

                with torch.no_grad():
                    for images, labels in tqdm(val_loader, desc="  Validating"):
                        images = _normalise(images.to(device))
                        labels = labels.to(device)
                        B, _, H, W = images.shape

                        # Resize to crop_size for forward pass
                        resized = F.interpolate(
                            images,
                            size=(self.crop_size, self.crop_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        logits = model(resized)
                        logits = F.interpolate(
                            logits,
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        )

                        val_loss = F.cross_entropy(
                            logits,
                            labels,
                            ignore_index=ignore_index,
                        )
                        val_loss_sum += val_loss.item() * B
                        val_count += B

                        # mIoU computation
                        preds = logits.argmax(dim=1)
                        mask = labels != ignore_index
                        for c in range(self._num_classes):
                            pred_c = (preds == c) & mask
                            label_c = (labels == c) & mask
                            total_intersection[c] += (pred_c & label_c).sum()
                            total_union[c] += (pred_c | label_c).sum()

                val_loss = val_loss_sum / max(val_count, 1)
                writer.add_scalar("val/loss", val_loss, epoch + 1)

                # mIoU (ignore classes with no union)
                valid = total_union > 0
                if valid.any():
                    iou = total_intersection[valid] / total_union[valid]
                    miou = iou.mean().item()
                else:
                    miou = 0.0
                writer.add_scalar("val/mIoU", miou, epoch + 1)

                print(f"  val_loss={val_loss:.4f}  val_mIoU={miou:.4f}")

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
        """Single forward pass → softmax probabilities + entropy.

        Resizes input to ``crop_size`` for the forward pass, then
        upsamples logits back to the original resolution.
        """
        assert self._model is not None, "Model not initialised. Call train() or load() first."
        device = images.device
        self._model.to(device)
        self._model.eval()

        B, _, H, W = images.shape

        with torch.no_grad():
            # Resize to crop_size
            resized = F.interpolate(
                images,
                size=(self.crop_size, self.crop_size),
                mode="bilinear",
                align_corners=False,
            )
            resized = _normalise(resized)
            logits = self._model(resized)  # (B, C, crop, crop)

            # Upsample back to original resolution
            logits = F.interpolate(
                logits,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

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

    def save(self, directory: Path) -> None:
        """Save only the MLP head weights + num_classes (~2.4MB)."""
        assert self._model is not None, "Model not initialised. Call train() before save()."
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "head_state_dict": self._model.head.state_dict(),
                "num_classes": self._num_classes,
            },
            directory / "checkpoint.pt",
        )

    def load(self, directory: Path) -> None:
        """Load head weights + num_classes (backbone re-loaded from hub)."""
        checkpoint = torch.load(
            directory / "checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        self._num_classes = checkpoint["num_classes"]
        self._model = self._build_model(self._num_classes)
        self._model.head.load_state_dict(checkpoint["head_state_dict"])


# -- Variants via inheritance ---------------------------------------------

class PastelDINOv2_N20(PastelDINOv2):
    """PASTEL ViT-L/14 with 20 training samples."""

    num_train_samples = 20
