import csv
import os
import lightning as L

class BestMetricsCallback(L.Callback):
    """
    Callback to collect all metrics for the best epoch measured by maximum val_R2Score.
    Saves results to CSV with metrics, property name, and run name.
    """

    def __init__(self, run_name: str, csv_path: str = "results.csv", classification: bool = False):
        """
        Args:
            run_name: Name of the current run
            csv_path: Path to save the CSV file
        """
        super().__init__()
        self.run_name = run_name
        self.csv_path = csv_path
        self.best_score = 1000.0 if not classification else -1.0
        self.current_epoch = 0
        self.best_metrics = {}
        self.classification = classification
        self.lead_metric = 'val_BinaryAUROC' if classification else 'val_MeanAbsoluteError'
        
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        current_metrics = trainer.callback_metrics

        if self.lead_metric in current_metrics:
            current_score = current_metrics[self.lead_metric].item()
            current_epoch = trainer.current_epoch
            if self.classification:
                # For classification, higher is better
                if current_score > self.best_score:
                    self.best_score = current_score
                    pl_module.best_metric = self.best_score # type: ignore
                    self.current_epoch = current_epoch
                    self.best_metrics = {k: v.item() if hasattr(v, 'item') else v 
                                       for k, v in current_metrics.items()}
            else:
                # For regression, lower is better
                if current_score < self.best_score:
                    self.best_score = current_score
                    pl_module.best_metric = self.best_score # type: ignore
                    self.current_epoch = current_epoch
                    self.best_metrics = {k: v.item() if hasattr(v, 'item') else v 
                                    for k, v in current_metrics.items()}
                
    def on_fit_end(self, trainer, pl_module) -> None:
        if not self.best_metrics:
            return
            
        row_data = {
            'run_name': self.run_name,
            'best_epoch': self.current_epoch,
            **self.best_metrics
        }
        
        file_exists = os.path.exists(self.csv_path)
        
        with open(self.csv_path, 'a', newline='') as csvfile:
            fieldnames = list(row_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row_data)
