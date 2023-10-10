import argparse

from bayestme.common import InferenceType


def add_deconvolution_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed value for random number generator."
    )
    parser.add_argument(
        "--inference-type",
        type=InferenceType,
        choices=list(InferenceType),
        default=InferenceType.SVI,
        help="Method for conducting inference.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        help="Number of cell types to deconvolve into.",
        default=None,
    )
    parser.add_argument(
        "--spatial-smoothing-parameter",
        type=float,
        help="Spatial smoothing parameter (referred to as lambda in paper)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        help="Number of samples from the posterior distribution or variational family.",
        default=100,
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

    mcmc = parser.add_argument_group(
        "MCMC", "MCMC Arguments (only used when --inference-type MCMC is specified)"
    )
    mcmc.add_argument(
        "--n-burn",
        type=int,
        help="Number of burn-in samples",
        default=1000,
    )
    mcmc.add_argument(
        "--n-thin",
        type=int,
        help="Thinning factor for sampling",
        default=10,
    )
    mcmc.add_argument(
        "--background-noise",
        help="Turn background noise on",
        action="store_true",
        default=False,
    )
    mcmc.add_argument(
        "--lda-initialization",
        help="Turn LDA Initialization on",
        action="store_true",
        default=False,
    )
    svi = parser.add_argument_group(
        "SVI", "SVI Arguments (only used when --inference-type SVI is specified)"
    )

    svi.add_argument(
        "--n-svi-steps",
        type=int,
        help="Number of steps for fitting variational family",
        default=10_000,
    )
    svi.add_argument(
        "--use-spatial-guide",
        help="Use spatial guide (variational family with spatial priors) for SVI",
        action="store_true",
        default=False,
    )

    return parser
