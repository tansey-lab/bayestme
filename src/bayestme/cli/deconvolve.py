import argparse
import logging

import numpy as np

import bayestme.data
import bayestme.log_config
import bayestme
import bayestme.expression_truth

from bayestme import data
from bayestme.mcmc import deconvolution
from bayestme.common import InferenceType

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
    parser.add_argument("--n-gene", type=int, help="number of genes")
    parser.add_argument(
        "--n-components",
        type=int,
        help="Number of cell types, expected to be determined from cross validation.",
        default=None,
    )
    parser.add_argument(
        "--lam2",
        type=float,
        help="Smoothness parameter, this tuning parameter expected to be determined"
        "from cross validation.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        help="Number of samples from the posterior distribution.",
        default=100,
    )
    parser.add_argument(
        "--n-burn", type=int, help="Number of burn-in samples", default=1000
    )
    parser.add_argument(
        "--n-thin", type=int, help="Thinning factor for sampling", default=10
    )
    parser.add_argument("--random-seed", type=int, help="Random seed", default=0)
    parser.add_argument(
        "--background-noise",
        help="Turn background noise on",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--lda-initialization",
        help="Turn LDA Initialization on",
        action="store_true",
        default=False,
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
        "--inference-type",
        type=InferenceType,
        choices=list(InferenceType),
        default=InferenceType.MCMC,
        help="Method for conducting inference.",
    )
    bayestme.log_config.add_logging_args(parser)
    return parser


def main():
    args = get_parser().parse_args()
    bayestme.log_config.configure_logging(args)

    dataset: data.SpatialExpressionDataset = data.SpatialExpressionDataset.read_h5(
        args.adata
    )

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

    rng = np.random.default_rng(seed=args.random_seed)

    results: data.DeconvolutionResult = deconvolution.deconvolve(
        reads=dataset.reads,
        edges=dataset.edges,
        n_gene=args.n_gene,
        n_components=n_components,
        lam2=args.lam2,
        n_samples=args.n_samples,
        n_burnin=args.n_burn,
        n_thin=args.n_thin,
        bkg=args.background_noise,
        lda=args.lda_initialization,
        expression_truth=expression_truth,
        rng=rng,
    )

    results.save(args.output)

    bayestme.data.add_deconvolution_results_to_dataset(stdata=dataset, result=results)

    if args.inplace:
        dataset.save(args.adata)
    else:
        dataset.save(args.adata_output)
