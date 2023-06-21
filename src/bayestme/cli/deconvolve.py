import argparse
import logging

import bayestme
import bayestme.cli.common
import bayestme.data
import bayestme.expression_truth
import bayestme.log_config
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

    dataset: data.SpatialExpressionDataset = data.SpatialExpressionDataset.read_h5(
        args.adata
    )

    rng = create_rng(args.seed)

    if args.expression_truth:
        expression_truth_samples = []
        for fn in args.expression_truth:
            expression_truth_samples.append(
                bayestme.expression_truth.load_expression_truth(dataset, fn)
            )
        n_components = expression_truth_samples[0].shape[0]

        if not len(set([x.shape for x in expression_truth_samples])) == 1:
            raise RuntimeError(
                "Multiple expression truth arrays were provided, and they have different dimensions. "
                "Please ensure --expression-truth arguments are correct."
            )

        expression_truth = bayestme.expression_truth.combine_multiple_expression_truth(
            expression_truth_samples
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
        mcmc_n_burn=args.n_burn,
        mcmc_n_thin=args.n_thin,
        n_svi_steps=args.n_svi_steps,
        background_noise=args.background_noise,
        lda_initialization=args.lda_initialization,
        expression_truth=expression_truth,
        inference_type=args.inference_type,
        use_spatial_guide=args.use_spatial_guide,
        rng=rng,
    )

    results.save(args.output)

    bayestme.data.add_deconvolution_results_to_dataset(stdata=dataset, result=results)

    if args.inplace:
        dataset.save(args.adata)
    else:
        dataset.save(args.adata_output)
