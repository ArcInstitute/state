import argparse as ap

from ._infer import add_arguments_infer, run_tx_infer
from ._predict import add_arguments_predict, run_tx_predict
from ._preprocess_train import add_arguments_preprocess_train, run_tx_preprocess_train
from ._sort import add_arguments_sort, run_tx_sort
from ._train import add_arguments_train, run_tx_train

__all__ = [
    "run_tx_train",
    "run_tx_predict",
    "run_tx_infer",
    "run_tx_preprocess_train",
    "run_tx_sort",
    "add_arguments_tx",
]


def add_arguments_tx(parser: ap.ArgumentParser):
    """"""
    subparsers = parser.add_subparsers(required=True, dest="subcommand")
    add_arguments_train(subparsers.add_parser("train", add_help=False))
    add_arguments_predict(subparsers.add_parser("predict"))
    add_arguments_infer(subparsers.add_parser("infer"))
    add_arguments_preprocess_train(subparsers.add_parser("preprocess_train"))
    add_arguments_sort(subparsers.add_parser("sort"))
