import argparse as ap
from hydra import initialize, compose
from omegaconf import DictConfig
from ._cli import (
    add_arguments_sets,
    add_arguments_state,
    run_sets_predict,
    run_sets_train,
    run_sets_infer,
    run_state_embed,
    run_state_train,
)


def get_args() -> tuple[ap.Namespace, list[str]]:
    """Parse known args and return remaining args for Hydra overrides"""
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="command")
    add_arguments_state(subparsers.add_parser("state"))
    add_arguments_sets(subparsers.add_parser("sets"))

    # Use parse_known_args to get both known args and remaining args
    return parser.parse_args()


def load_hydra_config(method: str, overrides: list[str] = None) -> DictConfig:
    """Load Hydra config with optional overrides"""
    if overrides is None:
        overrides = []

    # Initialize Hydra with the path to your configs directory
    # Adjust the path based on where this file is relative to configs/
    with initialize(version_base=None, config_path="configs"):
        match method:
            case "state":
                cfg = compose(config_name="state-defaults", overrides=overrides)
            case "sets":
                cfg = compose(config_name="config", overrides=overrides)
            case _:
                raise ValueError(f"Unknown method: {method}")
    return cfg


def main():
    args = get_args()

    match args.command:
        case "state":
            match args.subcommand:
                case "train":
                    cfg = load_hydra_config("state", args.hydra_overrides)
                    run_state_train(cfg, args)
                case "embed":
                    run_state_embed(args)
        case "sets":
            match args.subcommand:
                case "train":
                    # Load Hydra config with overrides for sets training
                    cfg = load_hydra_config("sets", args.hydra_overrides)
                    run_sets_train(cfg)
                case "predict":
                    # For now, predict uses argparse and not hydra
                    run_sets_predict(args)
                case "infer":
                    # Run inference using argparse, similar to predict
                    run_sets_infer(args)


if __name__ == "__main__":
    main()
