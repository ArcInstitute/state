import argparse as ap

from ._fit import add_arguments_fit, run_emb_fit
from ._transform import add_arguments_transform, run_emb_transform
from ._query import add_arguments_query, run_emb_query

__all__ = ["run_emb_fit", "run_emb_transform", "run_emb_query", "add_arguments_emb"]


def add_arguments_emb(parser: ap.ArgumentParser):
    """"""
    subparsers = parser.add_subparsers(required=True, dest="subcommand")
    add_arguments_fit(subparsers.add_parser("fit"))
    add_arguments_transform(subparsers.add_parser("transform"))
    add_arguments_query(subparsers.add_parser("query"))
