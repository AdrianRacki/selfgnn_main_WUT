from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from augmentors import Compose, EdgeRemoving, FeatureMasking
from dataset import SelfGraphDataset
from losses import barlow_twins_loss

DATA_ROOT = "/Users/adrianracki/Desktop/Projects/selfgnn_main/data"


class SelfGBT(L.LightningModule):
    """Class for SelfGBT model using PyTorch Lightning using ThreeLayerGAT encoder and Barlow Twins loss."""

    def __init__(
        self, params: dict[str, Any], encoder: torch.nn.Module | None = None
    ) -> None:
        """
        Initializes the SelfGBT model.

        Args:
            params (dict[str, Any]): Details in train_model function.
            encoder (torch.nn.Module | None): Pre-trained encoder model. Default is None and will use ThreeLayerGAT with params.
        """
        super().__init__()

        # Define encoder
        self.encoder = encoder
        # Unpack hyperparameters
        self.lr = params["lr"]
        self.weight_decay = params["weight_decay"]
        self.er_ratio = params["er_ratio"]
        self.fm_ratio = params["fm_ratio"]
        self.gamma = params["gamma"]
        # Save hyperparameters
        self.save_hyperparameters(params)
        # Define loss function
        self.loss = barlow_twins_loss

    def forward(self, graph):
        pass

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

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(  # noqa: F811
            optimizer, gamma=self.gamma
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


def train_encoder(
    params: dict[str, Any], encoder: torch.nn.Module | None = None
) -> L.LightningModule:
    """
    Trains the SelfGBT model with the given hyperparameters.

    Args:
        params (dict[str, Any]): Hyperparameters for the model.
            - node_dim (int): Input feature dimension i.e. number of node features. Must be defined if encoder is not provided.
            - hidden_dim (int): Hidden layers dimension. Must be defined if encoder is not provided.
            - out_dim (int): Output feature dimension. Input dimension for the predictor model. Must be defined if encoder is not provided.
            - edge_dim (int): Edge feature dimension i.e. number of edge features. Must be defined if encoder is not provided.
            - num_heads (int): Number of attention heads. Must be defined if encoder is not provided.
            - lr (float): Learning rate.
            - weight_decay (float): Weight decay for the optimizer.
            - er_ratio (float): Edge removing ratio.
            - fm_ratio (float): Feature masking ratio.
            - max_epochs (int): Maximum number of epochs.
            - batch_size (int): Batch size.
            - gamma (float): Gamma for the learning rate scheduler.

    Returns:
        L.LightningModule: Trained SelfGBT model.
    """
    # Data and model preparation
    dataset = SelfGraphDataset(DATA_ROOT)
    data = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)
    model = SelfGBT(params, encoder)
    # Callbacks
    logger = CSVLogger(save_dir="lightning_logs", name="ETest_runs")
    es = EarlyStopping("train_loss", patience=5, mode="min")
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{train_loss:.2f}",
        monitor="train_loss",
        save_top_k=2,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # Trainer
    trainer = L.Trainer(
        max_epochs=params["max_epochs"],
        deterministic=False,
        logger=logger,
        log_every_n_steps=50,
        accelerator="cpu",
        callbacks=[es, checkpoint_callback, lr_monitor],
    )

    trainer.fit(model, data)

    return model
