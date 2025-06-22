import argparse as ap

from ._predict import add_arguments_predict, run_sets_predict
from ._train import add_arguments_train, run_sets_train
from ._infer import add_arguments_infer, run_sets_infer

__all__ = ["run_sets_train", "run_sets_predict", "run_sets_infer", "add_arguments_sets"]


def add_arguments_sets(parser: ap.ArgumentParser):
    """"""
    subparsers = parser.add_subparsers(required=True, dest="subcommand")
    add_arguments_train(subparsers.add_parser("train"))
    add_arguments_predict(subparsers.add_parser("predict"))
    add_arguments_infer(subparsers.add_parser("infer"))
