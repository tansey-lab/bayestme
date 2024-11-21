import argparse

from bayestme.common import InferenceType


def add_deconvolution_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed value for random number generator."
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
        help="Matched scRNA data in h5ad format, will be used to enforce a prior on celltypes and expression.",
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
    parser.add_argument(
        "--n-svi-steps",
        type=int,
        help="Number of steps for fitting variational family",
        default=20_000,
    )
    parser.add_argument(
        "--use-spatial-guide",
        help="Use spatial guide (variational family with spatial priors) for SVI",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser
