import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics import MetricCollection
import math

# TODO: add prediction step
class GraphPredictor(L.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.batch_size = config.data.datamodule.batch_size
        self.model: torch.nn.Module = instantiate(config.model, _recursive_=False)
        self.optimizer = instantiate(
            config.trainer.optimizer, params=self.model.parameters()
        )
        self.scheduler = instantiate(config.trainer.scheduler, optimizer=self.optimizer)
        self.loss_fn = instantiate(config.trainer.loss)
        metrics = MetricCollection(
            [instantiate(metric) for metric in config.metrics.values()]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        
        # Log the number of parameters in the model and config
        k_params = sum(p.numel() for p in self.model.parameters()) / 1000
        config.trainer.model_k_params = math.ceil(k_params)
        self.save_hyperparameters(config)
        
    def step(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        x = self(batch)
        loss = self.loss_fn(x, batch.y.squeeze())
        return x, loss

    def forward(self, batch) -> torch.Tensor:
        x = self.model(batch)
        return x[0]

    def training_step(self, batch) -> torch.Tensor:
        x, loss = self.step(batch)
        output = self.train_metrics(x, batch.y.squeeze())
        self.log_dict(output, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, batch_size=self.batch_size
        )
        return loss

    def validation_step(self, batch) -> torch.Tensor:
        x, loss = self.step(batch)
        output = self.valid_metrics(x, batch.y.squeeze())
        self.log_dict(output, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log("val_loss", loss, batch_size=self.batch_size)
        return loss
    
    def get_mol_embedding(self, batch) -> torch.Tensor:
        """
        Get the molecular embedding from the model.
        """
        self.model.eval()
        with torch.no_grad():
            x = self.model(batch)
        return x[1]

    def configure_optimizers(self):  # type: ignore
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
