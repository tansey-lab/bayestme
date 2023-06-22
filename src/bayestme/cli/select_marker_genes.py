import argparse
import logging

import bayestme.log_config
import bayestme.marker_genes
from bayestme import data

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Perform marker gene selection")
    parser.add_argument("--adata", type=str, help="Input file, AnnData in h5 format")
    parser.add_argument(
        "--adata-output",
        type=str,
        help="A new AnnData in h5 format created with the deconvolution summary results "
        "appended.",
    )
    parser.add_argument(
        "-i",
        "--inplace",
        default=False,
        action="store_true",
        help="If provided, append deconvolution summary results to the --adata archive in place",
    )
    parser.add_argument(
        "--deconvolution-result",
        type=str,
        help="Input file, DeconvolutionResult in h5 format",
    )
    parser.add_argument(
        "--n-marker-genes",
        type=int,
        default=5,
        help="Maximum number of marker genes per cell type.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha cutoff for choosing marker genes.",
    )
    parser.add_argument(
        "--marker-gene-method",
        type=bayestme.marker_genes.MarkerGeneMethod,
        choices=list(bayestme.marker_genes.MarkerGeneMethod),
        default=bayestme.marker_genes.MarkerGeneMethod.TIGHT,
        help="Method for choosing marker genes.",
    )
    bayestme.log_config.add_logging_args(parser)

    return parser


def main():
    args = get_parser().parse_args()
    bayestme.log_config.configure_logging(args)
    logger.info("select_marker_genes called with arguments: {}".format(args))

    stdata = data.SpatialExpressionDataset.read_h5(args.adata)
    deconvolution_result = data.DeconvolutionResult.read_h5(args.deconvolution_result)

    marker_genes = bayestme.marker_genes.select_marker_genes(
        deconvolution_result=deconvolution_result,
        n_marker=args.n_marker_genes,
        alpha=args.alpha,
        method=args.marker_gene_method,
    )

    bayestme.marker_genes.add_marker_gene_results_to_dataset(
        stdata=stdata, marker_genes=marker_genes
    )

    if args.inplace:
        stdata.save(args.adata)
    else:
        stdata.save(args.adata_output)
