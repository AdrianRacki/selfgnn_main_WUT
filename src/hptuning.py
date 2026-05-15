import optuna
from hydra.utils import instantiate

from config_utils import load_config
from module import GraphPredictor

EXPERIMENT_NAME = "empty"
DIRECTION = "minimize"


def objective(trial: optuna.trial.Trial) -> float:

    # Hyperparameter search space
    lr = trial.suggest_float("trainer.optimizer.lr", 0.0001, 0.01, log=True)
    weight_decay = trial.suggest_float("trainer.optimizer.weight_decay", 1e-5, 0.001, log=True)
    huber_delta = trial.suggest_float("loss.delta", 0.1, 0.3, step=0.05)

    overrides = [
        "model=GIN",
        "trainer.trainer.max_epochs=50",
        f"trainer.optimizer.lr={lr}",
        f"trainer.optimizer.weight_decay={weight_decay}",
        f"trainer.loss.delta={huber_delta}",
        "run_name=Hptuning",
    ]

    config = load_config(experiment_name=EXPERIMENT_NAME, overrides=overrides)
    dataset = instantiate(config.data.dataset)
    datamodule = instantiate(config.data.datamodule, dataset=dataset)
    datamodule.setup()
    module = GraphPredictor(config)
    callbacks = list(instantiate(config.callbacks).values())
    trainer = instantiate(config.trainer.trainer, callbacks=callbacks, enable_checkpointing=False)
    trainer.fit(module, datamodule.train_dataloader(), datamodule.val_dataloader())
    return module.best_metric


if __name__ == "__main__":
    study = optuna.create_study(study_name="Hptuning", load_if_exists=True, direction=DIRECTION, storage="sqlite:///optuna_study.db")
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    print(study.best_value)
