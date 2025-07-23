import argparse as ap

from ._fit import add_arguments_fit, run_emb_fit
from ._transform import add_arguments_transform, run_emb_transform
from ._query import add_arguments_query, run_emb_query
from .._utils import CustomFormatter

__all__ = ["run_emb_fit", "run_emb_transform", "run_emb_query", "add_arguments_emb"]


def add_arguments_emb(parser: ap.ArgumentParser):
    """Add embedding commands to the parser"""
    subparsers = parser.add_subparsers(required=True, dest="subcommand")

    # fit
    desc = """description:
  Fit an embedding model to a dataset."""
    add_arguments_fit(subparsers.add_parser("fit", description=desc, formatter_class=CustomFormatter))

    # transform
    desc = """description:
  Transform a dataset into an embedding.
  You can also create or add embeddings to a LanceDB database."""
    add_arguments_transform(subparsers.add_parser("transform", description=desc, formatter_class=CustomFormatter))

    # query
    desc = """description:
  Query a LanceDB database for similar cells."""
    add_arguments_query(subparsers.add_parser("query", description=desc, formatter_class=CustomFormatter))
