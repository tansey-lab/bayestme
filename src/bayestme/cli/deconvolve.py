import argparse
import logging

from bayestme import data, deconvolution

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='Deconvolve data')
parser.add_argument('--input', type=str,
                    help='Input SpatialExpressionDataset in h5 format, '
                         'expected to be bleed corrected')
parser.add_argument('--output', type=str,
                    help='Path where DeconvolutionResult will be written h5 format')
parser.add_argument('--n-gene', type=int,
                    help='number of genes')
parser.add_argument('--n-components', type=int,
                    help='Number of cell types, expected to be determined from cross validation.')
parser.add_argument('--lam2', type=int,
                    help='Smoothness parameter, this tuning parameter expected to be determined'
                         'from cross validation.')
parser.add_argument('--n-samples', type=int,
                    help='Number of samples from the posterior distribution.',
                    default=100)
parser.add_argument('--n-burnin', type=int,
                    help='Number of burn-in samples',
                    default=1000)
parser.add_argument('--n-thin', type=int,
                    help='Thinning factor for sampling',
                    default=10)
parser.add_argument('--random-seed', type=int,
                    help='Random seed',
                    default=0)
parser.add_argument('--bkg',
                    help='Turn background noise on',
                    action='store_true',
                    default=False)
parser.add_argument('--lda',
                    help='Turn LDA Initialization on',
                    action='store_true',
                    default=False)


def main():
    args = parser.parse_args()

    dataset: data.SpatialExpressionDataset = data.SpatialExpressionDataset.read_h5(args.input)

    results: data.DeconvolutionResult = deconvolution.deconvolve(
        reads=dataset.reads,
        edges=dataset.edges,
        n_gene=args.n_gene,
        n_components=args.n_components,
        lam2=args.lam2,
        n_samples=args.n_samples,
        n_burnin=args.n_burnin,
        n_thin=args.n_thin,
        random_seed=args.random_seed,
        bkg=args.bkg,
        lda=args.lda
    )

    results.save(args.output)
