import argparse
import logging

import anndata
import pandas
import bayestme.log_config
import bayestme.plot.deconvolution
from bayestme import data
from bayestme.expression_truth import load_expression_truth

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Plot deconvolution results")
    parser.add_argument(
        "--adata",
        type=str,
        help="Input file, AnnData in h5 format. Expected to be annotated with deconvolution results.",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory.")
    parser.add_argument(
        "--cell-type-names",
        default=None,
        help="A comma separated list of cell type names to use for plots."
        'For example --cell-type-names "type 1, type 2, type 3"',
    )
    parser.add_argument(
        "--expression-truth",
        help="Use expression ground truth from one or matched scRNA datasets.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--reference-scrna-celltype-column",
        help="The name of the column with celltype id in the matched scRNA anndata.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--reference-scrna-sample-column",
        help="The name of the column with sample id in the matched scRNA anndata.",
        type=str,
        default=None,
    )
    bayestme.log_config.add_logging_args(parser)

    return parser


def main():
    args = get_parser().parse_args()
    bayestme.log_config.configure_logging(args)

    stdata = data.SpatialExpressionDataset.read_h5(args.adata)

    if args.cell_type_names is not None:
        cell_type_names = [name.strip() for name in args.cell_type_names.split(",")]
    else:
        cell_type_names = None

    if args.expression_truth is not None:
        cell_type_names = sorted(
            anndata.read_h5ad(args.expression_truth)
            .obs[args.reference_scrna_celltype_column]
            .unique()
        )

    bayestme.plot.deconvolution.plot_deconvolution(
        stdata=stdata, output_dir=args.output_dir, cell_type_names=cell_type_names
    )
