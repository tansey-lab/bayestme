import argparse
import logging

import bayestme.logging
from bayestme import data

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert data from spaceranger to a SpatialExpressionDataset in h5 format"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file, a SpatialExpressionDataset in h5 format",
    )
    parser.add_argument("--input", type=str, help="Input spaceranger dir")
    bayestme.logging.add_logging_args(parser)

    return parser


def main():
    args = get_parser().parse_args()
    bayestme.logging.configure_logging(args)

    dataset = data.SpatialExpressionDataset.read_spaceranger(
        args.input, layout=data.Layout.HEX
    )

    dataset.save(args.output)
