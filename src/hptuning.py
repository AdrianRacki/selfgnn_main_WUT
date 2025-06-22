import optuna
from hydra.utils import instantiate

from config_utils import load_config
from module import GraphPredictor

def objective(trial: optuna.trial.Trial) -> float:

    # Hyperparameter search space
    lr = trial.suggest_float("trainer.optimizer.lr", 1e-4, 0.01, log=True)
    weight_decay = trial.suggest_float(
        "trainer.optimizer.weight_decay", 1e-4, 0.01, log=True
    )
    batch_size = trial.suggest_float(
        "data.datamodule.batch_size", 16, 64, step=4
    )

    overrides = [
        f"trainer.optimizer.lr={lr}",
        f"trainer.optimizer.weight_decay={weight_decay}",
        f"data.datamodule.batch_size={int(batch_size)}",
        "run_name=Hptuning",
    ]
    
    config = load_config(experiment_name="mp_base", overrides=overrides)
    results = []
    for seed in range(42, 45):
        print(f"Running trial {trial.number} with seed {seed}")
        dataset = instantiate(config.data.dataset)
        datamodule = instantiate(config.data.datamodule, dataset=dataset, seed = int(seed))
        datamodule.setup()
        module = GraphPredictor(config)
        callbacks = list(instantiate(config.callbacks).values())
        trainer = instantiate(config.trainer.trainer, callbacks=callbacks, enable_checkpointing=False)
        trainer.fit(module, datamodule.train_dataloader(), datamodule.val_dataloader())
        results.append(trainer.callback_metrics["val_MeanAbsoluteError"].item())
    return sum(results) / len(results)

if __name__ == "__main__":
    study = optuna.create_study(study_name="Hptuning", load_if_exists=True, direction="minimize", storage="sqlite:///optuna_study.db")
    study.optimize(objective, n_trials=50)
    print(study.best_params)
    print(study.best_value)