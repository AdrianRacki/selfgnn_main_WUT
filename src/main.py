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

from dataset import LabeledGraphDataset


def main(
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
            - gamma (float): Gamma for the learning rate scheduler.

    Returns:
        L.LightningModule: Trained Predictor model.
    """
    # Data and model preparation
    model = Predictor(encoder, params)
    dataset = LabeledGraphDataset(DATA_ROOT).shuffle()
    train_dataset = dataset[: int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8) :]
    train_data = DataLoader(
        train_dataset,  # type: ignore
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_data = DataLoader(
        val_dataset,  # type: ignore
        batch_size=params["batch_size"],
        shuffle=False,
        drop_last=True,
    )
    # Callbacks
    logger = CSVLogger(save_dir="lightning_logs", name="PTest_runs")
    es = EarlyStopping("val_r2", patience=10, mode="max")
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{val_r2:.2f}",
        monitor="val_r2",
        save_top_k=2,
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # Trainer
    trainer = L.Trainer(
        max_epochs=params["max_epochs"],
        deterministic=False,
        logger=logger,
        log_every_n_steps=10,
        accelerator="cpu",
        callbacks=[es, checkpoint_callback, lr_monitor],
    )

    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)

    return model
