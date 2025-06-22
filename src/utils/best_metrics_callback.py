import csv
import os
import lightning as L

class BestMetricsCallback(L.Callback):
    """
    Callback to collect all metrics for the best epoch measured by maximum val_R2Score.
    Saves results to CSV with metrics, property name, and run name.
    """
    
    def __init__(self, run_name: str, csv_path: str = "results.csv"):
        """
        Args:
            run_name: Name of the current run
            csv_path: Path to save the CSV file
        """
        super().__init__()
        self.run_name = run_name
        self.csv_path = csv_path
        self.best_val_r2score = float('-inf')
        self.best_metrics = {}
        
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        current_metrics = trainer.callback_metrics
        
        if 'val_R2Score' in current_metrics:
            current_val_r2score = current_metrics['val_R2Score'].item()
            
            if current_val_r2score > self.best_val_r2score:
                self.best_val_r2score = current_val_r2score
                self.best_metrics = {k: v.item() if hasattr(v, 'item') else v 
                                   for k, v in current_metrics.items()}
                
    def on_fit_end(self, trainer, pl_module) -> None:
        if not self.best_metrics:
            return
            
        row_data = {
            'run_name': self.run_name,
            **self.best_metrics
        }
        
        file_exists = os.path.exists(self.csv_path)
        
        with open(self.csv_path, 'a', newline='') as csvfile:
            fieldnames = list(row_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row_data)
