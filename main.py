import cellseg
from omegaconf import DictConfig
import hydra
import mlflow


def log_params_recursive(cfg, prefix):
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
    print(results)
    return results[cfg.target]


if __name__ == "__main__":
    main()
