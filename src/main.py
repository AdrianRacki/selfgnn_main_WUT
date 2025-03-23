from hydra.utils import instantiate

from config_utils import load_config
from module import GraphPredictor


def main(overrides: list[str] | None = None):
    print("Running main")
    print("Loading config")
    config = load_config(overrides=overrides)
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


if __name__ == "__main__":
    main()
