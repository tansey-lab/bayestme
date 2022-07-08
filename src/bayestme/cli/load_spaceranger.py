import argparse
import logging

from bayestme import data

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Convert data from spaceranger to a SpatialExpressionDataset in h5 format')
parser.add_argument('--output', type=str,
                    help='Output file, a SpatialExpressionDataset in h5 format')
parser.add_argument('--input', type=str,
                    help='Input spaceranger dir')


def main():
    args = parser.parse_args()

    dataset = data.SpatialExpressionDataset.read_spaceranger(args.input, layout=data.Layout.HEX)

    dataset.save(args.output)
