"""PyTorch Lightning module and data module for the course's CORINE task.

Every bug listed below is from the 2025/26 lab5_1 / lab5_2 notebooks and is fixed
here, with the fix named in a comment so a student reading this can find the
corresponding cell in the old code.

1. **Scheduler stepped per batch against ``T_max`` in epochs.**
   ``{"optimizer": opt, "lr_scheduler": CosineAnnealingLR(opt, T_max=self.max_epochs)}``
   -- Lightning's default interval is ``"step"``, so with 625 steps/epoch and
   ``T_max=100`` the LR completed ~625 full cosine cycles instead of annealing
   once. Fixed by returning ``interval="epoch"`` explicitly.
2. **No imbalance-aware metric was ever logged.** Only ``val_loss``/``val_acc``
   existed, so no ``ModelCheckpoint(monitor=...)`` could select for rare-class
   performance. ``val_macro_f1`` and ``val_balanced_acc`` are logged here.
3. **Sampler built from the wrong split.** lab5_2 cell 46 computed weights over one
   subset and applied them to another, leaving 920 training patches unreachable.
   Weights here are derived from ``train_idx`` only, and the length is asserted.
4. **Augmentation could touch val/test.** ``test_dataset`` read
   ``cfg["transforms"]["val"]``, so setting the val transform to the train transform
   augmented the test set. Train/val/test transforms are separate keys and test is
   asserted to be ``None``.
5. **Per-batch accuracy averaged over batches, not samples.** A 124-sample final
   batch counted the same as a 256-sample one. Metrics are torchmetrics objects
   accumulated over the epoch, with ``sync_dist=True`` so DDP does not report a
   rank-0-only shard.
6. **No seeding.** ``seed_everything`` appeared zero times.
7. **Class weights unnormalised**, so the loss scale differed between arms and
   ``train_loss`` was not comparable across the very comparison the lab asked for.
   Weights are normalised to mean 1.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

try:  # torch and lightning are optional extras: the rest of eo_course runs without
    import lightning as L
    from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
except Exception as e:  # pragma: no cover
    raise ImportError(
        "eo_course.training needs torch and lightning (installed with the course "
        f"requirements). Original error: {e}"
    ) from None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PatchDataset(Dataset):
    """Wrap in-memory ``(N, C, H, W)`` float32 patches and int64 targets.

    Parameters
    ----------
    transform : callable | None
        Applied only when this dataset is used for training. Pass ``None`` for
        val/test; the data module enforces that.
    """

    def __init__(self, x, y, transform=None):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if x.ndim != 4:
            raise ValueError(f"expected (N,C,H,W), got {x.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x has {x.shape[0]} samples, y has {y.shape[0]}")
        if y.min() < 0:
            raise ValueError("targets must already be remapped to 0..K-1; negative index present")
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i):
        x = self.x[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[i]

    @property
    def targets(self) -> torch.Tensor:
        return self.y


def random_flip(x: torch.Tensor) -> torch.Tensor:
    """Random horizontal + vertical flip. The only augmentation the course uses.

    Deliberately geometric only. Spectral jitter would change the physical meaning
    of a band and is a separate, explicitly-chosen ablation, not a default.
    """
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[-1])
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[-2])
    return x


# ---------------------------------------------------------------------------
# Class-imbalance strategies
# ---------------------------------------------------------------------------
def class_weights(counts, mode: str, clamp: float = 50.0) -> torch.Tensor:
    """Per-class loss weights, normalised to mean 1.

    Normalising matters: without it the weighted arms have a loss several orders of
    magnitude different from the unweighted arm, so the training curves the lab asks
    you to compare are not comparable.

    ``clamp`` caps the weight of the rarest class. With a 396:1 imbalance ratio
    (lab5_1's own printed number) a raw 1/n weight is ~400x the majority weight and
    the optimiser spends the epoch chasing single samples.
    """
    c = torch.as_tensor(counts, dtype=torch.float64)
    if c.ndim != 1 or c.numel() == 0:
        raise ValueError("counts must be a non-empty 1-D array")
    if (c < 0).any():
        raise ValueError("counts must be non-negative")

    if mode in (None, "none"):
        return torch.ones(c.numel(), dtype=torch.float32)

    present = c > 0
    w = torch.ones_like(c)
    if mode in ("inverse_freq", "weighted_sampler"):
        w[present] = 1.0 / c[present]
    elif mode in ("sqrt_inverse_freq", "sqrt"):
        w[present] = 1.0 / torch.sqrt(c[present])
    else:
        raise ValueError(f"unknown imbalance mode {mode!r}")

    w = torch.clamp(w, max=float(clamp))
    w = w / w.mean()  # mean 1 keeps loss scale comparable across arms
    return w.to(torch.float32)


class FocalLoss(torch.nn.Module):
    """Focal loss, gamma=2 by default.

    Down-weights easy examples so the gradient concentrates on hard ones -- a
    different mechanism from reweighting, which is why it is its own arm.
    """

    def __init__(self, weight=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.register_buffer("weight", None if weight is None else weight.float())

    def forward(self, logits, target):
        ce = torch.nn.functional.cross_entropy(
            logits, target, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def make_criterion(imbalance: str | None, weights: torch.Tensor | None):
    if imbalance == "focal":
        return FocalLoss(weight=weights, gamma=2.0)
    return torch.nn.CrossEntropyLoss(weight=weights)


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------
class CorineDataModule(L.LightningDataModule):
    """Splits come from a :class:`~eo_course.splits.SplitManifest`, never from here.

    The 2025/26 DataModule recomputed its split on every ``setup()`` call and relied
    on a hard-coded ``seed=42`` default, so if the seed changed the "test" subset was
    drawn from a different permutation of the same pool -- direct train/test overlap.
    Indices are now passed in and cached, and the module refuses to invent them.
    """

    def __init__(
        self,
        x,
        y,
        manifest,
        batch_size: int = 256,
        imbalance: str | None = None,
        augment_train: bool = False,
        num_workers: int = 2,
        seed: int = 0,
    ):
        super().__init__()
        # save_hyperparameters() takes `ignore`, not a dict of overrides. The big
        # arrays and the manifest are excluded; the split hash is recorded separately
        # because it is the thing the grader recomputes.
        self.save_hyperparameters(ignore=["x", "y", "manifest"])
        self.hparams["split_manifest_hash"] = manifest.manifest_hash
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.manifest = manifest
        self.batch_size = int(batch_size)
        self.imbalance = imbalance
        self.augment_train = bool(augment_train)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.train_counts: np.ndarray | None = None
        self._datasets: dict = {}

    # -- construction ----------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        m = self.manifest
        # Train transform only if requested; val and test are unconditionally None.
        train_tf = random_flip if self.augment_train else None
        self._datasets = {
            "train": PatchDataset(self.x[m.train_idx], self.y[m.train_idx], transform=train_tf),
            "val": PatchDataset(self.x[m.val_idx], self.y[m.val_idx], transform=None),
            "test": PatchDataset(self.x[m.test_idx], self.y[m.test_idx], transform=None),
        }
        assert self._datasets["test"].transform is None, "test set must never be augmented"

        # Class counts from TRAIN ONLY. Feeding all-data counts into the loss is
        # leakage and also makes the weights wrong for the actual training pool.
        n_classes = int(self.y.max()) + 1
        self.train_counts = np.bincount(self.y[m.train_idx], minlength=n_classes).astype(np.int64)

    def _require(self, name: str) -> PatchDataset:
        if name not in self._datasets:
            raise RuntimeError(
                f"{name}_dataset is not built; setup() was not called for this stage. "
                "Use a Trainer, or call datamodule.setup() first."
            )
        return self._datasets[name]

    # -- samplers --------------------------------------------------------
    def _train_sampler(self) -> Sampler | None:
        if self.imbalance not in ("weighted_sampler",):
            return None
        ds = self._require("train")
        counts = np.bincount(ds.targets.numpy(), minlength=len(self.train_counts))
        w = class_weights(counts, "weighted_sampler")
        # Index the weights per-sample. Length must equal the dataset: lab5_2 built
        # 7,358 weights over an 8,278-sample dataset, so 920 patches were never
        # drawable and every weight was computed from the wrong subset.
        sample_w = w[ds.targets.numpy()]
        assert sample_w.numel() == len(ds), (
            f"sampler has {sample_w.numel()} weights for a dataset of {len(ds)}"
        )
        g = torch.Generator()
        g.manual_seed(self.seed)
        return WeightedRandomSampler(sample_w.double(), num_samples=len(ds), replacement=True, generator=g)

    # -- loaders ---------------------------------------------------------
    def train_dataloader(self):
        ds = self._require("train")
        sampler = self._train_sampler()
        kwargs = {}
        if sampler is None:
            # Passing both sampler= and shuffle=True raises; the sampler *is* the
            # shuffle when one is used.
            kwargs["shuffle"] = True
        return DataLoader(
            ds, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers,
            drop_last=False, persistent_workers=self.num_workers > 0, **kwargs,
        )

    def _eval_loader(self, name: str):
        ds = self._require(name)
        # batch_size=1 for evaluation makes thousands of sequential GPU launches and
        # hides the per-batch averaging bug; use a real batch size.
        return DataLoader(
            ds, batch_size=max(self.batch_size, 64), shuffle=False,
            num_workers=max(1, self.num_workers // 2), persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return self._eval_loader("val")

    def test_dataloader(self):
        return self._eval_loader("test")


# ---------------------------------------------------------------------------
# Model module
# ---------------------------------------------------------------------------
class SmallCNN(torch.nn.Module):
    """~150k parameters. Sized for the data, unlike lab5_2's 122 M-parameter VGG-16
    trained on 8,278 patches -- a memorisation signal the notebook never discussed.
    """

    def __init__(self, in_channels: int, n_classes: int, hidden: int = 32):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, hidden * 2, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(hidden * 2),
            torch.nn.ReLU(inplace=True),
        )
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden * 2, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class CorineModule(L.LightningModule):
    """Training step, and the metrics the checkpoint is allowed to monitor.

    ``val_macro_f1`` and ``val_balanced_acc`` are logged so ``ModelCheckpoint`` can
    select on them. Note that ``MulticlassAccuracy(average="macro")`` *is* balanced
    accuracy -- the mean of per-class recalls -- which is worth knowing before you
    reach for a third metric.
    """

    def __init__(
        self,
        n_classes: int,
        in_channels: int = 4,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        imbalance: str | None = None,
        train_counts=None,
        max_epochs: int = 50,
        hidden: int = 32,
    ):
        super().__init__()
        # `train_counts` is excluded from hyperparameters on purpose. It is a numpy
        # array, and since torch 2.6 `torch.load(weights_only=True)` is the default,
        # so a numpy array inside `hyper_parameters` makes load_from_checkpoint raise
        # UnpicklingError. Keep hparams scalar and store the counts as a buffer.
        self.save_hyperparameters(ignore=["train_counts"])
        self.n_classes = int(n_classes)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.imbalance = imbalance
        self.max_epochs = int(max_epochs)

        counts = np.ones(self.n_classes, dtype=np.int64) if train_counts is None else np.asarray(train_counts)
        self.register_buffer("train_counts", torch.as_tensor(counts, dtype=torch.long))
        self.register_buffer("w", class_weights(counts, "sqrt_inverse_freq" if imbalance == "focal" else imbalance))

        self.model = SmallCNN(in_channels, self.n_classes, hidden=hidden)
        self.criterion = make_criterion(imbalance, None if imbalance == "weighted_sampler" else self.w)

        common = {"num_classes": self.n_classes, "average": "macro", "multidim_average": "global"}
        self.val_acc_macro = MulticlassAccuracy(**common)
        self.val_f1_macro = MulticlassF1Score(**common, zero_division=0)

    # -- forward ---------------------------------------------------------
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc_macro.update(logits.argmax(1), y)
        self.val_f1_macro.update(logits.argmax(1), y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        # Computed once over the whole epoch, not averaged over batches.
        self.log("val_balanced_acc", self.val_acc_macro.compute(), prog_bar=True, sync_dist=True)
        self.log("val_macro_f1", self.val_f1_macro.compute(), prog_bar=True, sync_dist=True)
        self.val_acc_macro.reset()
        self.val_f1_macro.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.log("test_loss", self.criterion(self(x), y), on_step=False, on_epoch=True, sync_dist=True)

    # -- optimisation ----------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                # THE FIX. Default is "step", which with T_max in epochs turns the
                # schedule into hundreds of full cosine cycles.
                "interval": "epoch",
                "frequency": 1,
            },
        }
