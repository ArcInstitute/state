import argparse as ap
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from ._cli import (
    add_arguments_sets,
    add_arguments_state,
    run_sets_predict,
    run_sets_train,
    run_state_embed,
    run_state_train,
)

# Add this function and registration at the top of the file or inside run_sets_train before OmegaConf.to_container
def eval_resolver(expression_string: str):
    """
    A simple resolver that evaluates a Python expression string.
    WARNING: Use with caution if the expression_string can come from untrusted sources.
    """
    # The inner interpolations like '${model.kwargs.cell_set_len}'
    # should already be resolved by OmegaConf into their string values
    # before this resolver is called.
    # For example, if cell_set_len=512 and extra_tokens=1,
    # expression_string will be '512 + 1'.
    try:
        return eval(expression_string)
    except Exception as e:
        # Log or handle the error if evaluation fails
        print(f"Error evaluating expression string '{expression_string}': {e}")
        raise

if not OmegaConf.has_resolver("eval"): # Register only if not already registered
    OmegaConf.register_new_resolver("eval", eval_resolver, replace=True)

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


if __name__ == "__main__":
    main()
