import torch
from pathlib import Path
from torch.utils.data import random_split, Subset

SPLIT_DIR = Path("./outputs/splits")


def split_dataset(dataset, train_ratio=0.6, val_ratio=0.2, seed=42):

    SPLIT_DIR.mkdir(exist_ok=True)

    train_file = SPLIT_DIR / "train_indices.pt"
    val_file = SPLIT_DIR / "val_indices.pt"
    test_file = SPLIT_DIR / "test_indices.pt"

    # -----------------------------
    # Load existing split
    # -----------------------------
    if train_file.exists() and val_file.exists() and test_file.exists():

        train_indices = torch.load(train_file)
        val_indices = torch.load(val_file)
        test_indices = torch.load(test_file)

        print("Loaded existing dataset split.")

    # -----------------------------
    # Create new split
    # -----------------------------
    else:

        n = len(dataset)

        train_size = int(train_ratio * n)
        val_size = int(val_ratio * n)
        test_size = n - train_size - val_size

        train_subset, val_subset, test_subset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )

        train_indices = train_subset.indices
        val_indices = val_subset.indices
        test_indices = test_subset.indices

        torch.save(train_indices, train_file)
        torch.save(val_indices, val_file)
        torch.save(test_indices, test_file)

        print("Created and saved dataset split.")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset