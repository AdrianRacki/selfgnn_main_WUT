import math

import lightning as L
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics import MetricCollection

from model.base import OUTPUT_KEY


class GraphPredictor(L.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.best_metric = 0.0
        self.batch_size = config.data.datamodule.batch_size
        self.model: torch.nn.Module = instantiate(config.model)
        self.optimizer = instantiate(config.trainer.optimizer, params=self.model.parameters())
        scheduler_cfg = config.trainer.scheduler
        if hasattr(scheduler_cfg, "schedulers"):
            sub_schedulers = [instantiate(s, optimizer=self.optimizer) for s in scheduler_cfg.schedulers]
            self.scheduler = instantiate(scheduler_cfg, schedulers=sub_schedulers, optimizer=self.optimizer, _recursive_=False)
        else:
            self.scheduler = instantiate(scheduler_cfg, optimizer=self.optimizer)
        self.loss_fn = instantiate(config.trainer.loss)
        metrics = MetricCollection([instantiate(metric) for metric in config.metrics.values()])
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")

        # Log the number of parameters in the model and config
        k_params = sum(p.numel() for p in self.model.parameters()) / 1000
        config.trainer.model_k_params = math.ceil(k_params)
        self.save_hyperparameters(config)

        # Initialize lists for predictions and SMILES
        self.preds = []
        self.smiles = []
        self.y_true = []

    def step(self, batch) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        x = self(batch)
        loss = self.loss_fn(x[OUTPUT_KEY], batch.y.squeeze(), frequency=batch.frq)
        return x, loss

    def forward(self, batch) -> dict[str, torch.Tensor]:
        x = self.model(batch)
        return x

    def training_step(self, batch) -> torch.Tensor:
        x, loss = self.step(batch)
        output = self.train_metrics(x[OUTPUT_KEY], batch.y.squeeze())
        self.log_dict(output, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch) -> torch.Tensor:
        x, loss = self.step(batch)
        output = self.valid_metrics(x[OUTPUT_KEY], batch.y.squeeze())
        self.log_dict(output, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log("val_loss", loss, batch_size=self.batch_size)
        return loss

    def predict_step(self, batch) -> torch.Tensor:
        x = self(batch)
        return x[OUTPUT_KEY]

    def on_predict_batch_end(self, outputs, batch, batch_idx):  # type: ignore
        self.preds.extend(outputs)
        self.y_true.extend(batch.y.squeeze().tolist())
        self.smiles.extend(batch.smiles)

    def on_predict_end(self) -> None:
        df = pd.DataFrame(
            {
                "smiles": self.smiles,
                "value": [p.item() for p in self.preds],
                "y_true": self.y_true,
            }
        )
        df.to_csv(f"data/predictions/{self.config.run_name}_predictions.csv", index=False)

    def get_mol_embedding(self, batch) -> torch.Tensor:
        """
        Get the molecular embedding from the model.
        """
        self.model.eval()
        with torch.no_grad():
            x = self.model(batch)
        return x[OUTPUT_KEY]

    def configure_optimizers(self):  # type: ignore
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
