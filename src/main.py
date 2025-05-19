from hydra.utils import instantiate

from config_utils import load_config
from module import GraphPredictor
import argparse


def main(experiment_name: str, overrides: list[str] | None = None):
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
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    overrides = [f"run_name={args.run_name}"]
    main(experiment_name=args.experiment, overrides=overrides)
