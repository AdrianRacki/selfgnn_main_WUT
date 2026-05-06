import pytorch_lightning as pl
import torch
from torch.utils.data import random_split
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


class GNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        data_split: float = 0.8,
        seed: int = 42,
    ):
        """
        Initialize the GNN DataModule.

        Args:
            dataset: Graph dataset to use
            batch_size: Size of each batch
            data_split: Fraction of data to use for training (rest goes to validation)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_split = data_split
        self.seed = seed

    def setup(self, stage: str | None = None):
        torch.manual_seed(self.seed)
        train_size = int(len(self.dataset) * self.data_split)
        val_size = len(self.dataset) - train_size
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size], generator)

    def setup_n_folds(self, n_folds: int = 10, fold: int = 0):
        """Setup n-fold cross-validation.

        Args:
            n_folds: Number of folds for cross-validation (default: 10)
            fold: Current fold index (0 to n_folds-1) (default: 0)
        """
        torch.manual_seed(self.seed)

        if fold < 0 or fold >= n_folds:
            raise ValueError(f"Fold index must be between 0 and {n_folds - 1}, got {fold}")

        dataset_size = len(self.dataset)
        fold_size = dataset_size // n_folds
        remainder = dataset_size % n_folds

        fold_indices = []
        start_idx = 0
        for i in range(n_folds):
            current_fold_size = fold_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_fold_size
            fold_indices.append(range(start_idx, end_idx))
            start_idx = end_idx

        all_indices = torch.randperm(dataset_size)
        val_indices = [all_indices[i].item() for i in fold_indices[fold]]

        train_indices = []
        for i in range(n_folds):
            if i != fold:
                train_indices.extend([all_indices[j].item() for j in fold_indices[i]])

        self.train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(self.dataset, val_indices)  # type: ignore

        print(f"Created {n_folds}-fold CV split: fold {fold} as validation set")
        print(f"Training samples: {len(self.train_dataset)}, Validation samples: {len(self.val_dataset)}")

    def train_dataloader(self):
        """Return DataLoader for training data."""
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        """Return DataLoader for validation data."""
        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        """Return DataLoader for test data.

        Note: This uses the validation set as test set if no separate test set is provided.
        """
        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )

    def predict_dataloader(self):
        """Return DataLoader for prediction data. Return full dataset."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
