"""
dataset.py
----------
CIFAR-10 data-loading utilities.

Augmentation pipeline (training):
  RandomCrop(32, padding=4) → RandomHorizontalFlip → ColorJitter → Normalize

Validation/Test pipeline:
  Normalize only

Both use the canonical CIFAR-10 channel statistics.
"""

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T


# ---- canonical CIFAR-10 statistics (computed over the entire training set) ----
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_transforms(augment: bool = True):
    """Return train/val transform pair."""
    normalize = T.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        normalize,
    ]) if augment else T.Compose([T.ToTensor(), normalize])

    val_tf = T.Compose([T.ToTensor(), normalize])
    return train_tf, val_tf


def get_dataloaders(
    data_dir: str   = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    val_split: float = 0.1,
    augment: bool = True,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Download (if needed) and return (train_loader, val_loader, test_loader).

    The training set is split 90/10 into train/val by default.
    The 10 k-sample test set is always separate.

    Parameters
    ----------
    data_dir    : root directory for dataset storage
    batch_size  : samples per mini-batch
    num_workers : DataLoader worker processes
    val_split   : fraction of train set used for validation
    augment     : apply training augmentation to the train split
    pin_memory  : pin memory tensors (recommended when using CUDA)

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_tf, val_tf = get_transforms(augment=augment)

    # Download full training set twice (different transforms) so we can split
    full_train   = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    full_train_v = torchvision.datasets.CIFAR10(data_dir, train=True,  download=False, transform=val_tf)
    test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True,  transform=val_tf)

    n_val   = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val

    # Deterministic split
    generator = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(
        range(len(full_train)), [n_train, n_val], generator=generator
    )
    train_idx = list(train_idx)
    val_idx   = list(val_idx)

    train_set = torch.utils.data.Subset(full_train,   train_idx)
    val_set   = torch.utils.data.Subset(full_train_v, val_idx)

    common_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader = DataLoader(train_set, shuffle=True,  **common_kw)
    val_loader   = DataLoader(val_set,   shuffle=False, **common_kw)
    test_loader  = DataLoader(test_dataset, shuffle=False, **common_kw)

    return train_loader, val_loader, test_loader


def get_classes() -> list[str]:
    return [
        "airplane", "automobile", "bird", "cat",
        "deer", "dog", "frog", "horse", "ship", "truck",
    ]
