from typing import Any

import lightning as L
import mlflow
import torch
from lightning.pytorch.loggers import CSVLogger
from torch_geometric.data import Data

from augmentors import Compose, EdgeRemoving, FeatureMasking
from encoders import ThreeLayerGAT
from loss import barlow_twins_loss


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


def train_model(params: dict[str, Any], data: Any) -> L.LightningModule:
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
        data (Any): Data for training the model, usually in the form of a PyTorch Geometric DataLoader.

    Returns:
        L.LightningModule: Trained SelfGBT model.
    """
    L.seed_everything(47)

    model = SelfGBT(params)
    logger = CSVLogger(save_dir="lightning_logs", name="Test_runs")
    trainer = L.Trainer(
        max_epochs=params["max_epochs"], deterministic=True, logger=logger, accelerator="cpu"
    )

    trainer.fit(model, data)

    return model
