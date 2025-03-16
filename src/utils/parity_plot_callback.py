from typing import Any, List, Tuple

import lightning as L
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch


class ParityPlotCallback(L.Callback):
    """
    PyTorch Lightning callback that creates and logs parity plots to MLflow.
    """

    def __init__(self, plot_every_n_epochs: int = 1):
        """
        Initialize the callback.

        Args:
            plot_every_n_epochs: Generate and log parity plot every N epochs
        """
        super().__init__()
        self.plot_every_n_epochs = plot_every_n_epochs
        self.val_predictions: List[torch.Tensor] = []
        self.val_targets: List[torch.Tensor] = []

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect predictions and targets after each validation batch."""
        # Get predictions from the batch
        with torch.no_grad():
            y_pred = pl_module(batch)
            y_true = batch.y.squeeze()

            self.val_predictions.append(y_pred.detach().cpu())
            self.val_targets.append(y_true.detach().cpu())

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Create and log parity plot at the end of validation."""
        current_epoch = trainer.current_epoch

        # Check if we should create a plot in this epoch
        if current_epoch % self.plot_every_n_epochs != 0:
            # Clear the collected data to save memory
            self.val_predictions = []
            self.val_targets = []
            return

        # Concatenate all predictions and targets
        if len(self.val_predictions) > 0 and len(self.val_targets) > 0:
            all_preds = torch.cat(self.val_predictions, dim=0)
            all_targets = torch.cat(self.val_targets, dim=0)

            # Create parity plot
            fig = create_parity_plot(
                all_targets,
                all_preds,
                title=f"Parity Plot - Epoch {current_epoch}",
                xlabel="True Values",
                ylabel="Predicted Values",
            )

            # Log to MLflow
            if trainer.logger and hasattr(trainer.logger, "experiment"):
                # Log the plot as an artifact
                mlflow.log_figure(fig, f"parity_plot_epoch_{current_epoch}.png")

                # Close the figure to prevent memory leaks
                plt.close(fig)

            # Clear the collected data
            self.val_predictions = []
            self.val_targets = []


def create_parity_plot(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    title: str = "Parity Plot",
    xlabel: str = "True Values",
    ylabel: str = "Predicted Values",
    figsize: Tuple[int, int] = (10, 10),
) -> plt.Figure: # type: ignore
    """
    Create a parity plot comparing true values vs predicted values.

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size as (width, height)

    Returns:
        matplotlib Figure object
    """
    # Convert tensors to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy() # type: ignore
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy() # type: ignore

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot predictions as scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5)

    # Add perfect prediction line (y=x)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    # Add some margin
    margin = (max_val - min_val) * 0.1
    min_val -= margin
    max_val += margin

    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")

    # Set limits
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    return fig
