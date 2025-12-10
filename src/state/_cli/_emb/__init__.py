import argparse as ap

from ._fit import add_arguments_fit, run_emb_fit
from ._transform import add_arguments_transform, run_emb_transform
from ._query import add_arguments_query, run_emb_query
from ._preprocess import add_arguments_preprocess, run_emb_preprocess
from ._eval import add_arguments_eval, run_emb_eval

__all__ = [
    "run_emb_fit",
    "run_emb_transform",
    "run_emb_query",
    "run_emb_preprocess",
    "run_emb_eval",
    "add_arguments_emb",
]


def add_arguments_emb(parser: ap.ArgumentParser):
    """"""
    subparsers = parser.add_subparsers(required=True, dest="subcommand")
    add_arguments_fit(subparsers.add_parser("fit"))
    add_arguments_transform(subparsers.add_parser("transform"))
    add_arguments_query(subparsers.add_parser("query"))
    add_arguments_preprocess(
        subparsers.add_parser("preprocess", help="Preprocess datasets and create embedding profiles")
    )
    add_arguments_eval(subparsers.add_parser("eval", help="Evaluate embeddings"))
