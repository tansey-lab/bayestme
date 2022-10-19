import argparse
import logging
import bayestme.logging

import numpy as np
import os

from bayestme import data, spatial_expression


MODEL_DUMP_PATH = 'sde_model_dump.h5'


def get_parser():
    parser = argparse.ArgumentParser(description='Detect spatial differential expression patterns')
    parser.add_argument('--deconvolve-results',
                        type=str,
                        help='DeconvolutionResult in h5 format')
    parser.add_argument('--adata',
                        type=str,
                        help='AnnData in h5 format')
    parser.add_argument('--output', type=str,
                        help='Path to store SpatialDifferentialExpressionResult in h5 format')
    parser.add_argument('--n-cell-min',
                        type=int,
                        default=5,
                        help='Only consider spots where there are at least <n_cell_min> cells of a given type, '
                             'as determined by the deconvolution results.')
    parser.add_argument('--n-spatial-patterns', type=int,
                        help='Number of spatial patterns.')
    parser.add_argument('--n-samples', type=int,
                        default=100,
                        help='Number of samples from the posterior distribution.')
    parser.add_argument('--n-burn', type=int,
                        default=1000,
                        help='Number of burn-in samples')
    parser.add_argument('--n-thin', type=int,
                        default=2,
                        help='Thinning factor for sampling')
    parser.add_argument('--simple',
                        action='store_true',
                        default=False,
                        help='Simpler model for sampling spatial differential expression posterior')
    parser.add_argument('--alpha0', type=int,
                        help='Alpha0 tuning parameter. Defaults to 10',
                        default=10)
    parser.add_argument('--prior-var', type=float,
                        help='Prior var tuning parameter. Defaults to 100.0',
                        default=100.0)
    parser.add_argument('--lam2', type=int,
                        help='Smoothness parameter, this tuning parameter expected to be determined '
                             'from cross validation.',
                        default=1)
    parser.add_argument('--n-gene', type=int,
                        help='Number of genes to consider for detecting spatial programs,'
                             ' if this number is less than the total number of genes the top N'
                             ' by spatial variance will be selected')
    bayestme.logging.add_logging_args(parser)

    return parser


def main():
    args = get_parser().parse_args()
    bayestme.logging.configure_logging(args)

    dataset: data.SpatialExpressionDataset = data.SpatialExpressionDataset.read_h5(
        args.adata)

    deconvolve_results: data.DeconvolutionResult = data.DeconvolutionResult.read_h5(
        args.deconvolve_results)

    alpha = np.ones(args.n_spatial_patterns + 1)
    alpha[0] = args.alpha0
    alpha[1:] = 1 / args.n_spatial_patterns

    n_nodes = dataset.n_spot_in
    n_signals = dataset.n_gene
    prior_vars = np.repeat(args.prior_var, 2)

    sde = spatial_expression.SpatialDifferentialExpression(
        n_cell_types=deconvolve_results.n_components,
        n_spatial_patterns=args.n_spatial_patterns,
        n_nodes=n_nodes,
        n_signals=n_signals,
        edges=dataset.edges,
        alpha=alpha,
        prior_vars=prior_vars,
        lam2=args.lam2
    )

    sde.initialize()

    sampler_state_path = os.path.join(os.path.dirname(args.output), MODEL_DUMP_PATH)

    try:
        results = spatial_expression.run_spatial_expression(
            sde=sde,
            deconvolve_results=deconvolve_results,
            n_samples=args.n_samples,
            n_burn=args.n_burn,
            n_thin=args.n_thin,
            n_cell_min=args.n_cell_min,
            simple=args.simple
        )
    except Exception as e:
        logging.exception(f'Exception raised during posterior sampling, will save current sampler state '
                          f'to {sampler_state_path} for debugging.')
        if sde.has_checkpoint:
            sde.reset_to_checkpoint()

        sde.get_state().save(sampler_state_path)

        raise e

    results.save(args.output)
