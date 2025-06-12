from hydra.utils import instantiate

from config_utils import load_config
from module import GraphPredictor
from datamodule import GNNDataModule
import argparse
import tqdm

def main(experiment_name: str, overrides: list[str] | None = None) -> tuple[GraphPredictor, GNNDataModule]:
    print("Running main")
    print("Loading config")
    config = load_config(experiment_name = experiment_name, overrides=overrides)
    print("Loading data")
    dataset = instantiate(config.data.dataset)
    datamodule = instantiate(config.data.datamodule, dataset=dataset)
    datamodule.setup()
    print("Creating model")
    module = GraphPredictor(config)
    callbacks = list(instantiate(config.callbacks).values())
    print("Preparing trainer")
    trainer = instantiate(config.trainer.trainer, callbacks=callbacks)
    print("Training model")
    trainer.fit(module, datamodule.train_dataloader(), datamodule.val_dataloader())
    return module, datamodule

def main_nfolds(experiment_name: str, n_folds: int, overrides: list[str] | None = None):
    for fold in tqdm.tqdm(range(n_folds), desc="Running k-fold"):
        fold_overrides = [f"run_name={experiment_name}_fold_{fold}"]
        if overrides:
            fold_overrides.extend(overrides)
        config = load_config(experiment_name=experiment_name, overrides=fold_overrides)
        dataset = instantiate(config.data.dataset)
        datamodule = instantiate(config.data.datamodule, dataset=dataset)
        datamodule.setup_n_folds(n_folds=n_folds, fold=fold)
        module = GraphPredictor(config)
        callbacks = list(instantiate(config.callbacks).values())
        trainer = instantiate(config.trainer.trainer, callbacks=callbacks)
        trainer.fit(module, datamodule.train_dataloader(), datamodule.val_dataloader())

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--run_name",
        type=str,
        default="default_run",
        help="Name of the run",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment main config name",
    )
    parser.add_argument(
        "--run_nfolds",
        type=int,
        default=0,
        help="Number of folds for cross-validation. 0 means no cross-validation",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.run_nfolds > 0:
        main_nfolds(
            experiment_name=args.experiment,
            n_folds=args.run_nfolds,
        )
    else:
        overrides = [f"run_name={args.run_name}"]
        main(experiment_name=args.experiment, overrides=overrides)
