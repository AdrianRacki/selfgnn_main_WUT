import optuna
from hydra.utils import instantiate

from config_utils import load_config
from module import GraphPredictor

def objective(trial: optuna.trial.Trial) -> float:
    # Hyperparameter search space
    lr = trial.suggest_float("trainer.optimizer.lr", 1e-5, 0.01, log=True)
    weight_decay = trial.suggest_float(
        "trainer.optimizer.weight_decay", 1e-6, 0.01, log=True
    )
    loss_target = trial.suggest_categorical(
        "trainer.loss._target_",
        [
            "torch.nn.MSELoss",
            "torch.nn.L1Loss",
            "torch.nn.SmoothL1Loss",
            "torch.nn.HuberLoss",
        ],
    )
    hidden_dim = trial.suggest_categorical("model.hidden_dim", [4, 8, 16, 32, 64])
    emb_size = trial.suggest_categorical("model.emb_size", [4, 8, 16, 32])
    num_layers = trial.suggest_categorical("model.num_layers", [1, 2, 3, 4, 6, 8])
    num_heads = trial.suggest_categorical("model.num_heads", [1, 2, 4])
    dropout_rate = trial.suggest_categorical("model.dropout_rate", [0.3, 0.4, 0.5])
    use_global_features = trial.suggest_categorical(
        "model.use_global_features", [True, False]
    )
    beta = trial.suggest_categorical("model.beta", [True, False])

    batch_size = trial.suggest_categorical(
        "data.datamodule.batch_size", [16, 32, 64]
    )

    # Convert to Hydra-style overrides
    overrides = [
        f"trainer.optimizer.lr={lr}",
        f"trainer.optimizer.weight_decay={weight_decay}",
        f"trainer.loss._target_={loss_target}",
        f"model.hidden_dim={hidden_dim}",
        f"model.emb_size={emb_size}",
        f"model.num_layers={num_layers}",
        f"model.num_heads={num_heads}",
        f"model.dropout_rate={dropout_rate}",
        f"model.use_global_features={use_global_features}",
        f"model.beta={beta}",
        f"data.datamodule.batch_size={batch_size}",
    ]
    
    config = load_config(overrides=overrides)
    dataset = instantiate(config.data.dataset)
    datamodule = instantiate(config.data.datamodule, dataset=dataset)
    datamodule.setup()
    module = GraphPredictor(config)
    callbacks = list(instantiate(config.callbacks).values())
    trainer = instantiate(config.trainer.trainer, callbacks=callbacks, enable_checkpointing=False)
    trainer.fit(module, datamodule.train_dataloader(), datamodule.val_dataloader())
    
    return trainer.callback_metrics["val_MeanAbsoluteError"].item()

if __name__ == "__main__":
    study = optuna.create_study(study_name="GNNhptuning", direction="minimize", storage="sqlite:///optuna_study.db")
    study.optimize(objective, n_trials=1000)
    print(study.best_params)
    print(study.best_value)