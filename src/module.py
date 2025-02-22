from typing import Any

import lightning as L
import torch
from sklearn.metrics import r2_score
from torch.nn import BatchNorm1d, Dropout, Linear, PReLU, Sequential
from torch_geometric.nn import global_add_pool as gap


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
        self.gamma = params["gamma"]

        # Freeze encoder weights
        if freeze_encoder is True:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Aggregation layer and predictor
        # self.aggr = SetTransformerAggregation(channels=in_dim, heads=2, dropout=0.2)
        self.predictor = Sequential(
            Linear(in_dim, hidden_size),
            BatchNorm1d(hidden_size),
            PReLU(),
            Dropout(0.2),
            Linear(hidden_size, int(hidden_size / 2)),
            BatchNorm1d(int(hidden_size / 2)),
            PReLU(),
            Dropout(0.2),
            Linear(int(hidden_size / 2), 1),
        )
        # Save hyperparameters
        self.save_hyperparameters(params)

    def step() -> None:
        pass

    def forward(self, graph) -> torch.Tensor:
        x = self.encoder(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        # x = self.aggr(x, graph.batch)
        x = gap(x, graph.batch)
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

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(  # noqa: F811
            optimizer, gamma=self.gamma
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
