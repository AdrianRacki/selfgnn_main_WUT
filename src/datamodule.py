import pytorch_lightning as pl
import torch
from torch.utils.data import random_split
from torch_geometric.data import Dataset
from typing import Optional
from torch_geometric.loader import DataLoader

class GNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        data_split: float = 0.8,
        seed: int = 42
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
        
    def setup(self, stage: Optional[str] = None):
        torch.manual_seed(self.seed)
        train_size = int(len(self.dataset) * self.data_split)
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, 
            [train_size, val_size]
        )
        
    def train_dataloader(self):
        """Return DataLoader for training data."""
        return DataLoader(
            self.train_dataset, # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
    
    def val_dataloader(self):
        """Return DataLoader for validation data."""
        return DataLoader(
            self.val_dataset, # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
    
    def test_dataloader(self):
        """Return DataLoader for test data.
        
        Note: This uses the validation set as test set if no separate test set is provided.
        """
        return DataLoader(
            self.val_dataset, # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )

