import cellseg
from omegaconf import DictConfig
import hydra


@hydra.main(version_base=None, config_path='conf', config_name="config.yaml")
def main(cfg: DictConfig) -> float:
    print(cfg)
    results = cellseg.train(cfg)
    return 0.0


if __name__ == "__main__":
    main()
