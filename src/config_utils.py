import hydra
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str = "config") -> DictConfig:
    # Making config_path absolute path
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg: DictConfig = hydra.compose("train.yaml")
    return cfg
