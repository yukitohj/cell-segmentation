import cellseg
from omegaconf import DictConfig
import hydra
import mlflow


def log_params_recursive(cfg: DictConfig, prefix: str = "") -> None:
    """入れ子になったコンフィグを再帰的に記録していきます

    Args:
        cfg (DictConfig): コンフィグ
        prefix (_type_): 入れ子になったときの親のラベル名
    """
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            log_params_recursive(value, prefix+key+".")
        else:
            mlflow.log_param(prefix+key, value)


@hydra.main(version_base=None, config_path='conf', config_name="config.yaml")
def main(cfg: DictConfig) -> float:
    with mlflow.start_run():
        log_params_recursive(cfg, "")
        results = cellseg.train(cfg)
    return results[cfg.target]


if __name__ == "__main__":
    main()
