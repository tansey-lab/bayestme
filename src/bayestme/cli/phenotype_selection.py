import argparse
import os

from bayestme import phenotype_selection, data
import bayestme.logging


def get_parser():
    parser = argparse.ArgumentParser(
        description="Select values for number of cell types and "
        "lambda smoothing parameter "
        "via k-fold cross-validation."
    )
    parser.add_argument("--adata", type=str, help="Input file, AnnData in h5 format")
    parser.add_argument(
        "--job-index",
        type=int,
        default=None,
        help="Run only this job index, "
        "suitable for running the sampling in parallel across many machines",
    )
    parser.add_argument(
        "--n-fold",
        type=int,
        default=5,
        help="Number of times to run k-fold cross-validation.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=15,
        help="Split dataset into k consecutive folds for each instance of k-fold cross-validation",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples from the posterior distribution.",
    )
    parser.add_argument(
        "--n-burn", type=int, default=2000, help="Number of burn-in samples"
    )
    parser.add_argument(
        "--n-thin", type=int, default=5, help="Thinning factor for sampling"
    )
    parser.add_argument(
        "--n-gene",
        type=int,
        default=1000,
        help="Use N top genes by standard deviation to model deconvolution. "
        "If this number is less than the total number of genes the top N"
        " by spatial variance will be selected",
    )
    parser.add_argument(
        "--n-components-min",
        type=int,
        default=2,
        help="Minimum number of cell types to try.",
    )
    parser.add_argument(
        "--n-components-max",
        type=int,
        default=12,
        help="Maximum number of cell types to try.",
    )
    parser.add_argument(
        "--lambda-values",
        type=float,
        action="append",
        help="Potential values of the lambda smoothing parameter to try. "
        "Defaults to (1, 1e1, 1e2, 1e3, 1e4, 1e5)",
    )
    parser.add_argument(
        "--max-ncell",
        type=int,
        default=120,
        help="Maximum cell count within a spot to model.",
    )
    parser.add_argument("--background-noise", default=False, action="store_true")
    parser.add_argument("--lda-initialization", default=False, action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory. N new files will be saved in this directory, "
        "where N is the number of cross-validation jobs.",
    )
    bayestme.logging.add_logging_args(parser)

    return parser


DEFAULT_LAMBDAS = (1, 1e1, 1e2, 1e3, 1e4, 1e5)


def main():
    args = get_parser().parse_args()
    bayestme.logging.configure_logging(args)

    stdata = data.SpatialExpressionDataset.read_h5(args.adata)

    all_jobs = [
        _
        for _ in enumerate(
            phenotype_selection.get_phenotype_selection_parameters_for_folds(
                stdata=stdata,
                n_fold=args.n_fold,
                n_splits=args.n_splits,
                lams=DEFAULT_LAMBDAS if not args.lambda_values else args.lambda_values,
                n_components_min=args.n_components_min,
                n_components_max=args.n_components_max,
            )
        )
    ]

    # We're just going to run one parameter set of the grid search if the --job-index
    # parameter is passed
    if args.job_index is not None:
        all_jobs = [all_jobs[args.job_index]]

    for job_index, (lam, n_components_for_job, mask, fold_number) in all_jobs:
        result: data.PhenotypeSelectionResult = (
            phenotype_selection.run_phenotype_selection_single_job(
                lam=lam,
                n_components=n_components_for_job,
                mask=mask,
                fold_number=fold_number,
                stdata=stdata,
                n_samples=args.n_samples,
                n_burn=args.n_burn,
                n_thin=args.n_thin,
                max_ncell=args.max_ncell,
                n_gene=args.n_gene,
                background_noise=args.background_noise,
                lda_initialization=args.lda_initialization,
            )
        )

        result.save(os.path.join(args.output_dir, "fold_{}.h5ad".format(job_index)))
