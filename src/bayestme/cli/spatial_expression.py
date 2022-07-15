import argparse
from bayestme import data, spatial_expression

parser = argparse.ArgumentParser(description='Detect spatial differential expression patterns')
parser.add_argument('--deconvolve-results',
                    type=str,
                    help='DeconvolutionResult in h5 format')
parser.add_argument('--dataset',
                    type=str,
                    help='SpatialExpressionDataset in h5 format')
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


def main():
    args = parser.parse_args()

    dataset: data.SpatialExpressionDataset = data.SpatialExpressionDataset.read_h5(
        args.dataset)

    deconvolve_results: data.DeconvolutionResult = data.DeconvolutionResult.read_h5(
        args.deconvolve_results)

    results = spatial_expression.run_spatial_expression(
        dataset=dataset,
        deconvolve_results=deconvolve_results,
        n_spatial_patterns=args.n_spatial_patterns,
        n_samples=args.n_samples,
        n_burn=args.n_burn,
        n_thin=args.n_thin,
        n_cell_min=args.n_cell_min,
        alpha0=args.alpha0,
        prior_var=args.prior_var,
        lam2=args.lam2,
        simple=args.simple
    )

    results.save(args.output)
