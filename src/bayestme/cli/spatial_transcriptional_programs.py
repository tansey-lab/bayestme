import argparse
import logging
import os
import pandas

import bayestme
import bayestme.cli.common
import bayestme.data
import bayestme.expression_truth
import bayestme.log_config
import bayestme.plot.deconvolution
from bayestme import data
from bayestme import spatial_transcriptional_programs
from bayestme.common import create_rng

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Learn spatial transcriptional programs"
    )
    parser.add_argument("--adata", type=str, help="Input AnnData in h5 format")
    parser.add_argument(
        "--deconvolution-result", type=str, help="DeconvolutionResult in h5 format"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path where SpatialDifferentialExpressionResult will be written h5 format",
    )
    parser.add_argument(
        "--n-spatial-programs",
        type=int,
        help="Number of spatial programs per cell type to learn",
        default=5,
    )
    parser.add_argument(
        "--expression-truth",
        help="Use expression ground truth from one or matched samples that have been processed "
        "with the seurat companion scRNA fine mapping workflow. This flag can be provided multiple times"
        " for multiple matched samples.",
        type=str,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--cell-type-names",
        default=None,
        help="A comma separated list of cell type names to use for plots."
        'For example --cell-type-names "type 1, type 2, type 3"',
    )
    parser.add_argument(
        "--trend-filtering-lambda",
        type=float,
        help="Hyperparameter for spatial trend filtering",
        default=1.0,
    )
    parser.add_argument(
        "--lasso-lambda",
        type=float,
        help="Hyperparameter for L1 regularization",
        default=0.1,
    )
    parser.add_argument(
        "--gene-batch-size",
        type=int,
        help="Batch size of genes per iteration",
        default=50,
    )
    parser.add_argument(
        "--spot-batch-size",
        type=int,
        help="Batch size of spots per iteration",
        default=1000,
    )
    parser.add_argument(
        "--n-iter", type=int, help="Number of iterations", default=10_000
    )
    parser.add_argument(
        "--seed", type=int, help="Seed value for random number generator", default=None
    )
    bayestme.log_config.add_logging_args(parser)

    return parser


def main():
    args = get_parser().parse_args()
    bayestme.log_config.configure_logging(args)
    logger.info(
        "spatial_transcriptional_programs called with arguments: {}".format(args)
    )

    dataset: data.SpatialExpressionDataset = data.SpatialExpressionDataset.read_h5(
        args.adata
    )

    rng = create_rng(args.seed)

    deconvolution_results = data.DeconvolutionResult.read_h5(args.deconvolution_result)

    if args.cell_type_names is not None:
        cell_type_names = [name.strip() for name in args.cell_type_names.split(",")]
    else:
        cell_type_names = None

    if args.expression_truth is not None:
        cell_type_names = pandas.read_csv(
            args.expression_truth[0], index_col=0
        ).columns.tolist()

        # pad cell type names up to length stdata.n_cell_types
        i = 1
        while len(cell_type_names) < dataset.n_cell_types:
            cell_type_names.append(f"unknown_{i}")
            i += 1

    stp_results = spatial_transcriptional_programs.train(
        data=dataset,
        deconvolution_result=deconvolution_results,
        n_programs=args.n_spatial_programs,
        n_steps=args.n_iter,
        batchsize_spots=args.spot_batch_size,
        batchsize_genes=args.gene_batch_size,
        trend_filtering_lambda=args.trend_filtering_lambda,
        lasso_lambda=args.lasso_lambda,
        rng=rng,
    )
    logger.info("Saving SpatialDifferentialExpressionResult to {}".format(args.output))

    stp_results.save(args.output)

    plot_dir = os.path.join(os.path.dirname(args.output), "stp_plots")
    logger.info("Plotting spatial transcriptional programs to {}".format(plot_dir))

    os.makedirs(plot_dir, exist_ok=True)

    spatial_transcriptional_programs.plot_spatial_transcriptional_programs(
        data=dataset,
        stp=stp_results,
        output_dir=plot_dir,
        cell_type_names=cell_type_names,
    )
    spatial_transcriptional_programs.plot_loss(
        stp=stp_results,
        output_path=os.path.join(plot_dir, "training_loss.pdf"),
    )
    spatial_transcriptional_programs.plot_top_spatial_program_genes(
        data=dataset,
        stp=stp_results,
        output_dir=plot_dir,
        cell_type_names=cell_type_names,
    )
