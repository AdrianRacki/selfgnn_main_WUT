import hydra
from omegaconf import DictConfig, OmegaConf

def load_config(experiment_name: str, config_path: str = "config", overrides = None) -> DictConfig:
    try :
        OmegaConf.register_new_resolver("len", lambda x: len(x))
    except ValueError:
        pass
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg: DictConfig = hydra.compose(f"{experiment_name}.yaml", overrides=overrides)
    return cfg
