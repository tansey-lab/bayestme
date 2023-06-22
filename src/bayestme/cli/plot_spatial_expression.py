import argparse
import logging

import bayestme.log_config
from bayestme import data, spatial_expression

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Plot spatial differential expression results"
    )
    parser.add_argument("--adata", type=str, help="Input file, AnnData in h5 format")
    parser.add_argument(
        "--deconvolution-result",
        type=str,
        help="Input file, DeconvolutionResult in h5 format",
    )
    parser.add_argument(
        "--sde-result",
        type=str,
        help="Input file, SpatialDifferentialExpressionResult in h5 format",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--cell-type-names",
        default=None,
        help="A comma separated list of cell type names to use for plots."
        'For example --cell-type-names "type 1, type 2, type 3"',
    )
    parser.add_argument(
        "--moran-i-score-threshold",
        default=0.9,
        type=float,
        help="Moran I score threshold for selecting significant spatial patterns",
    )
    parser.add_argument(
        "--tissue-threshold",
        default=5,
        type=int,
        help="Only consider spots with greater than this many cells of type k for Moran's I "
        "calculation and cell correlation calculation",
    )
    parser.add_argument(
        "--gene-spatial-pattern-proportion-threshold",
        default=0.95,
        type=float,
        help="Only consider spatial patterns significant where "
        "greater than this proportion of spots are labeled with spatial "
        "pattern for at least one gene.",
    )
    bayestme.log_config.add_logging_args(parser)

    return parser


def main():
    args = get_parser().parse_args()
    bayestme.log_config.configure_logging(args)

    stdata = data.SpatialExpressionDataset.read_h5(args.adata)
    deconvolution_result = data.DeconvolutionResult.read_h5(args.deconvolution_result)
    sde_result = data.SpatialDifferentialExpressionResult.read_h5(args.sde_result)

    if args.cell_type_names is not None:
        cell_type_names = [name.strip() for name in args.cell_type_names.split(",")]
    else:
        cell_type_names = None

    spatial_expression.plot_significant_spatial_patterns(
        stdata=stdata,
        decon_result=deconvolution_result,
        sde_result=sde_result,
        output_dir=args.output_dir,
        cell_type_names=cell_type_names,
        moran_i_score_threshold=args.moran_i_score_threshold,
        tissue_threshold=args.tissue_threshold,
        gene_spatial_pattern_proportion_threshold=args.gene_spatial_pattern_proportion_threshold,
    )
