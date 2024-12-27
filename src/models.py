from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from augmentors import Compose, EdgeRemoving, FeatureMasking
from dataset import SelfGraphDataset
from encoders import ThreeLayerGAT
from loss import barlow_twins_loss

DATA_ROOT = "/Users/adrianracki/Desktop/Projects/selfgnn_main/data"


class SelfGBT(L.LightningModule):
    """Class for SelfGBT model using PyTorch Lightning using ThreeLayerGAT encoder and Barlow Twins loss."""

    def __init__(self, params) -> None:
        """
        Initializes the SelfGBT model.

        Args:
            params (dict[str, Any]): Details in train_model function.
        """
        super().__init__()

        # Unpack hyperparameters
        self.encoder = ThreeLayerGAT(
            params["node_dim"],
            params["hidden_dim"],
            params["out_dim"],
            params["edge_dim"],
            params["num_heads"],
        )
        self.lr = params["lr"]
        self.weight_decay = params["weight_decay"]
        self.er_ratio = params["er_ratio"]
        self.fm_ratio = params["fm_ratio"]
        self.save_hyperparameters(params)
        # Define loss function
        self.loss = barlow_twins_loss

    def forward(self, graph) -> torch.Tensor:
        x, edge_index, edge_attr, batch_index = (
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch,
        )
        return self.encoder(x, edge_index, edge_attr, batch_index)

    def training_step(self, graph) -> torch.Tensor:
        augmentor = Compose(
            [EdgeRemoving(self.er_ratio), FeatureMasking(self.fm_ratio)]
        )
        graph_a = augmentor(graph)
        graph_b = augmentor(graph)
        z_a = self(graph_a)
        z_b = self(graph_b)
        loss = self.loss(z_a, z_b)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, graph):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


def train_model(params: dict[str, Any]) -> L.LightningModule:
    """
    Trains the SelfGBT model with the given hyperparameters and data.

    Args:
        params (dict[str, Any]): Hyperparameters for the model.
            - node_dim (int): Input feature dimension.
            - hidden_dim (int): Hidden layer dimension.
            - out_dim (int): Output feature dimension.
            - edge_dim (int): Edge feature dimension.
            - num_heads (int): Number of attention heads.
            - lr (float): Learning rate.
            - weight_decay (float): Weight decay for the optimizer.
            - er_ratio (float): Edge removing ratio.
            - fm_ratio (float): Feature masking ratio.
            - max_epochs (int): Maximum number of epochs.
            - batch_size (int): Batch size.
        data (Any): Data for training the model, usually in the form of a PyTorch Geometric DataLoader.

    Returns:
        L.LightningModule: Trained SelfGBT model.
    """
    # Data and model preparation
    dataset = SelfGraphDataset(DATA_ROOT)
    data = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)
    model = SelfGBT(params)
    # Callbacks
    logger = CSVLogger(save_dir="lightning_logs", name="Test_runs")
    es = EarlyStopping("train_loss", patience=5, mode="min")
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{train_loss:.2f}",
        monitor="train_loss",
        save_top_k=2,
        mode="min",
    )
    # Trainer
    trainer = L.Trainer(
        max_epochs=params["max_epochs"],
        deterministic=False,
        logger=logger,
        accelerator="cpu",
        callbacks=[es,checkpoint_callback],
    )

    trainer.fit(model, data)

    return model
