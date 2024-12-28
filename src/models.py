from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import r2_score
from torch.nn import BatchNorm1d, Linear, PReLU, Sequential
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SetTransformerAggregation

from augmentors import Compose, EdgeRemoving, FeatureMasking
from dataset import LabeledGraphDataset, SelfGraphDataset
from encoders import ThreeLayerGAT
from loss import barlow_twins_loss

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
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = ThreeLayerGAT(
                params["node_dim"],
                params["hidden_dim"],
                params["out_dim"],
                params["edge_dim"],
                params["num_heads"],
            )
        # Unpack hyperparameters
        self.lr = params["lr"]
        self.weight_decay = params["weight_decay"]
        self.er_ratio = params["er_ratio"]
        self.fm_ratio = params["fm_ratio"]
        # Save hyperparameters
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


class Predictor(L.LightningModule):
    """
    A simple linear predictor model that can take pre-trained encoder.

    Args:
        encoder (torch.nn.Module): The encoder model. Likely a pre-trained model.
        params (dict[str, Any]): Details in train_predictor function.
    """

    def __init__(self, encoder: torch.nn.Module, params: dict[str, Any]) -> None:
        super().__init__()
        freeze_encoder = params.get("freeze_encoder", True)
        self.encoder = encoder
        self.lr = params["lr"]
        hidden_size = params["hidden_size"]
        in_dim = params["in_dim"]
        self.weight_decay = params["weight_decay"]
        self.loss = torch.nn.L1Loss()
        self.batch_size = params["batch_size"]

        # Freeze encoder weights
        if freeze_encoder is True:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Aggregation layer and predictor
        self.aggr = SetTransformerAggregation(channels=in_dim, heads=2)
        self.predictor = Sequential(
            Linear(in_dim, hidden_size),
            BatchNorm1d(hidden_size),
            PReLU(),
            Linear(hidden_size, int(hidden_size / 2)),
            BatchNorm1d(int(hidden_size / 2)),
            PReLU(),
            Linear(int(hidden_size / 2), 1),
        )
        # Save hyperparameters
        self.save_hyperparameters(params)

    def forward(self, graph) -> torch.Tensor:
        x = self.encoder(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        x = self.aggr(x, graph.batch)
        return self.predictor(x).squeeze()

    def training_step(self, graph) -> torch.Tensor:
        y = self(graph)
        loss = self.loss(y, (target := graph.y.float()))
        r2 = r2_score(target.numpy(), y.detach().numpy())
        self.log("train_loss", loss, batch_size=self.batch_size)
        self.log("train_r2", r2, batch_size=self.batch_size)  # type: ignore
        return loss

    def validation_step(self, graph) -> None:
        y = self(graph)
        loss = self.loss(y, (target := graph.y.float()))
        r2 = r2_score(target.numpy(), y.detach().numpy())
        self.log("val_loss", loss, batch_size=self.batch_size)
        self.log("val_r2", r2, batch_size=self.batch_size)  # type: ignore

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


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

    Returns:
        L.LightningModule: Trained SelfGBT model.
    """
    # Data and model preparation
    dataset = SelfGraphDataset(DATA_ROOT)
    data = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)
    model = SelfGBT(params, encoder)
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
        log_every_n_steps=50,
        accelerator="cpu",
        callbacks=[es, checkpoint_callback],
    )

    trainer.fit(model, data)

    return model


def train_predictor(
    params: dict[str, Any], encoder: torch.nn.Module
) -> L.LightningModule:
    """
    Trains the Predictor model with the given encoder, hyperparameters and data. 20% of the data is used for validation.

    Args:
        params (dict[str, Any]): Hyperparameters for the model.
            - lr (float): Learning rate.
            - hidden_size (int): Hidden layer dimension.
            - in_dim (int): Input feature dimension from encoder.
            - weight_decay (float): Weight decay for the optimizer.
            - freeze_encoder Optional(bool): Whether to freeze the encoder weights. Default is True.
            - max_epochs (int): Maximum number of epochs.
            - batch_size (int): Batch size.

    Returns:
        L.LightningModule: Trained Predictor model.
    """
    # Data and model preparation
    model = Predictor(encoder, params)
    dataset = LabeledGraphDataset(DATA_ROOT).shuffle()
    train_dataset = dataset[: int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8) :]
    train_data = DataLoader(
        train_dataset, # type: ignore
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,  
    )
    val_data = DataLoader(
        val_dataset, # type: ignore
        batch_size=params["batch_size"],
        shuffle=False,
        drop_last=True,  
    )
    # Callbacks
    logger = CSVLogger(save_dir="lightning_logs", name="PTest_runs")
    es = EarlyStopping("val_r2", patience=5, mode="max")
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{val_r2:.2f}",
        monitor="val_r2",
        save_top_k=2,
        mode="max",
    )
    # Trainer
    trainer = L.Trainer(
        max_epochs=params["max_epochs"],
        deterministic=False,
        logger=logger,
        log_every_n_steps=10,
        accelerator="cpu",
        callbacks=[es, checkpoint_callback],
    )

    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)

    return model
