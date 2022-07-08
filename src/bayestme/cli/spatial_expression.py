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
                    help='min number of cell types')
parser.add_argument('--n-spatial-patterns', type=int,
                    help='number of spatial patterns')
parser.add_argument('--n-samples', type=int,
                    default=100,
                    help='number of samples')
parser.add_argument('--n-burn', type=int,
                    default=1000,
                    help='burnin iterations')
parser.add_argument('--n-thin', type=int,
                    default=2,
                    help='thin iterations')
parser.add_argument('--simple',
                    action='store_true',
                    default=False,
                    help='simple mode')
parser.add_argument('--alpha0', type=int,
                    help='Alpha0 tuning parameter',
                    default=10)
parser.add_argument('--prior-var', type=float,
                    help='Prior var tuning parameter',
                    default=100.0)
parser.add_argument('--lam2', type=int,
                    help='lam2 tuning parameter',
                    default=1)
parser.add_argument('--n-gene', type=int,
                    help='number of genes')


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
