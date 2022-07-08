import argparse
import os

from bayestme import phenotype_selection, data

parser = argparse.ArgumentParser(description='Deconvolve data')
parser.add_argument('--stdata', type=str,
                    help='Input file, SpatialExpressionDataset in h5 format')
parser.add_argument('--fold-idx', type=int,
                    help='Run only this fold index, suitable for running the sampling in parallel across many machines')
parser.add_argument('--n-fold', type=int, default=5)
parser.add_argument('--n-splits', type=int, default=15)
parser.add_argument('--n-samples', type=int, default=100)
parser.add_argument('--n-burn', type=int, default=2000)
parser.add_argument('--n-thin', type=int, default=5)
parser.add_argument('--n-gene', type=int, default=1000)
parser.add_argument('--n-components-min', type=int, default=2)
parser.add_argument('--n-components-max', type=int, default=12)
parser.add_argument('--lambda-values',
                    type=float,
                    action='append')
parser.add_argument('--max-ncell', type=int, default=120)
parser.add_argument('--background-noise', default=False, action='store_true')
parser.add_argument('--lda-initialization', default=False, action='store_true')
parser.add_argument('--output-dir', type=str,
                    help='Output directory')
DEFAULT_LAMBDAS = (1, 1e1, 1e2, 1e3, 1e4, 1e5)


def main():
    args = parser.parse_args()

    stdata = data.SpatialExpressionDataset.read_h5(args.stdata)

    result: data.PhenotypeSelectionResult = phenotype_selection.run_phenotype_selection_single_fold(
        fold_idx=args.fold_idx,
        stdata=stdata,
        n_fold=args.n_fold,
        n_splits=args.n_splits,
        lams=DEFAULT_LAMBDAS if not args.lambda_values else args.lambda_values,
        n_components_min=args.n_components_min,
        n_components_max=args.n_components_max,
        n_samples=args.n_samples,
        n_burn=args.n_burn,
        n_thin=args.n_thin,
        max_ncell=args.max_ncell,
        n_gene=args.n_gene,
        background_noise=args.background_noise,
        lda_initialization=args.lda_initialization
    )

    result.save(os.path.join(args.output_dir, 'fold_{}.h5ad'.format(args.fold_idx)))
