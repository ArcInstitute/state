import argparse as ap

from ._infer import add_arguments_infer, run_tx_infer
from ._predict import add_arguments_predict, run_tx_predict
from ._train import add_arguments_train, run_tx_train
from .._utils import CustomFormatter

__all__ = ["run_tx_train", "run_tx_predict", "run_tx_infer", "add_arguments_tx"]


def add_arguments_tx(parser: ap.ArgumentParser):
    """Add transcriptomic commands to the parser"""
    subparsers = parser.add_subparsers(required=True, dest="subcommand")

    # Train
    desc = """description:
  Train a perturbation model using a Hydra configuration.
  Provide overrides to customize training, e.g.:
  `state tx train data.batch_size=32`"""
    add_arguments_train(
        subparsers.add_parser("train", description=desc, formatter_class=CustomFormatter)
    )

    # Predict
    desc = """description:
  Generate predictions from a trained model and optionally compute evaluation metrics."""
    add_arguments_predict(
        subparsers.add_parser("predict", description=desc, formatter_class=CustomFormatter)
    )

    # Infer
    desc = """description:
  Run inference on new samples using a trained model."""
    add_arguments_infer(
        subparsers.add_parser("infer", description=desc, formatter_class=CustomFormatter)
    )
