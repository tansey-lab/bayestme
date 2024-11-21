import argparse
import logging
import os

import anndata

import bayestme
import bayestme.cli.common
import bayestme.data
import bayestme.expression_truth
import bayestme.log_config
import bayestme.plot.deconvolution
from bayestme import data
from bayestme import deconvolution
from bayestme.common import create_rng

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Deconvolve data")
    parser.add_argument(
        "--adata",
        type=str,
        help="Input AnnData in h5 format, expected to be already bleed corrected",
    )
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
        "--output",
        type=str,
        help="Path where DeconvolutionResult will be written h5 format",
    )
    bayestme.cli.common.add_deconvolution_arguments(parser)
    bayestme.log_config.add_logging_args(parser)
    return parser


def main():
    args = get_parser().parse_args()
    bayestme.log_config.configure_logging(args)
    logger.info("deconvolution called with arguments: {}".format(args))

    dataset: data.SpatialExpressionDataset = data.SpatialExpressionDataset.read_h5(
        args.adata
    )

    rng = create_rng(args.seed)

    if args.expression_truth:
        ad = anndata.read_h5ad(args.expression_truth)

        expression_truth = (
            bayestme.expression_truth.calculate_celltype_profile_prior_from_adata(
                ad,
                dataset.gene_names,
                celltype_column=args.expression_truth_celltype_column,
                sample_column=args.expression_truth_sample_column,
            )
        )
    else:
        expression_truth = None
        n_components = args.n_components

    if n_components is None:
        raise RuntimeError(
            "--n-components not explicitly provided, and no expression truth provided."
        )

    results: data.DeconvolutionResult = deconvolution.sample_from_posterior(
        data=dataset,
        n_components=n_components,
        spatial_smoothing_parameter=args.spatial_smoothing_parameter,
        n_samples=args.n_samples,
        n_svi_steps=args.n_svi_steps,
        expression_truth=expression_truth,
        use_spatial_guide=args.use_spatial_guide,
        rng=rng,
    )

    results.save(args.output)

    if results.losses is not None:
        bayestme.plot.deconvolution.plot_loss(
            results,
            os.path.join(os.path.dirname(args.output), "deconvolution_loss.pdf"),
        )

    bayestme.data.add_deconvolution_results_to_dataset(stdata=dataset, result=results)

    if args.inplace:
        dataset.save(args.adata)
    else:
        dataset.save(args.adata_output)
