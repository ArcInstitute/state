import os
import sys
import logging
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Optional

sys.path.append("../../")
import vci.train.trainer as train

log = logging.getLogger(__name__)
# os.environ["NCCL_TIMEOUT"] = "36000"

# Custom command line resolver for hydra
import argparse


def main(config_path: Optional[str] = None):
    parser = argparse.ArgumentParser(description="VCI pretraining")
    parser.add_argument("--conf", type=str, help="Path to config YAML file")

    # First parse just the conf argument
    args, override_args = parser.parse_known_args()

    if not args.conf:
        parser.error("--conf argument is required")

    # Initialize hydra with the directory of the config file
    config_file = Path(args.conf)
    config_dir = str(config_file.parent)
    config_name = config_file.name

    # Initialize configuration
    with hydra.initialize_config_module(config_module=None, version_base=None):
        # Load the base configuration
        cfg = OmegaConf.load(args.conf)

        # Process the remaining command line arguments as overrides
        if override_args:
            overrides = OmegaConf.from_dotlist(override_args)
            cfg = OmegaConf.merge(cfg, overrides)

        # Execute the main logic
        run_with_config(cfg)


def run_with_config(cfg: DictConfig):
    if cfg.embeddings.current is None:
        log.error("Gene embeddings are required for training. Please set 'embeddings.current'")
        sys.exit(1)

    if cfg.dataset.current is None:
        log.error("Please set the desired dataset to 'dataset.current'")
        sys.exit(1)

    os.environ["MASTER_PORT"] = str(cfg.experiment.port)
    # WAR: Workaround for sbatch failing when --ntasks-per-node is set.
    # lightning expects this to be set.
    os.environ["SLURM_NTASKS_PER_NODE"] = str(cfg.experiment.num_gpus_per_node)

    log.info(f"*************** Training {cfg.experiment.name} ***************")
    log.info(OmegaConf.to_yaml(cfg))

    train.main(cfg)


if __name__ == "__main__":
    main()
