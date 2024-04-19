import argparse
import os

import bayestme.cli.common
import bayestme.log_config
import logging
from bayestme import phenotype_selection, data
from bayestme.common import create_rng


logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Select values for number of cell types and "
        "spatial smoothing parameter "
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
        "--spatial-smoothing-values",
        type=float,
        action="append",
        help="Potential values of the spatial smoothing parameter to try. "
        "Defaults to (1, 1e1, 1e2, 1e3, 1e4, 1e5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory. N new files will be saved in this directory, "
        "where N is the number of cross-validation jobs.",
    )
    bayestme.cli.common.add_deconvolution_arguments(parser)
    bayestme.log_config.add_logging_args(parser)

    return parser


DEFAULT_LAMBDAS = (1, 1e1, 1e2, 1e3, 1e4, 1e5)


def main():
    args = get_parser().parse_args()
    bayestme.log_config.configure_logging(args)
    logger.info("phenotype_selection called with arguments: {}".format(args))

    rng = create_rng(args.seed)

    stdata = data.SpatialExpressionDataset.read_h5(args.adata)

    all_jobs = [
        _
        for _ in enumerate(
            phenotype_selection.get_phenotype_selection_parameters_for_folds(
                stdata=stdata,
                n_fold=args.n_fold,
                n_splits=args.n_splits,
                lams=DEFAULT_LAMBDAS
                if not args.spatial_smoothing_values
                else args.spatial_smoothing_values,
                n_components_min=args.n_components_min,
                n_components_max=args.n_components_max,
            )
        )
    ]

    # We're just going to run one parameter set of the grid search if the --job-index
    # parameter is passed
    if args.job_index is not None:
        all_jobs = [all_jobs[args.job_index]]

    logger.info("Running {} jobs".format(len(all_jobs)))

    for job_index, (
        spatial_smoothing_param,
        n_components_for_job,
        mask,
        fold_number,
    ) in all_jobs:
        result: data.PhenotypeSelectionResult = (
            phenotype_selection.run_phenotype_selection_single_job(
                spatial_smoothing_parameter=spatial_smoothing_param,
                n_components=n_components_for_job,
                mask=mask,
                fold_number=fold_number,
                stdata=stdata,
                n_samples=args.n_samples,
                n_svi_steps=args.n_svi_steps,
                use_spatial_guide=args.use_spatial_guide,
                rng=rng,
            )
        )

        result.save(os.path.join(args.output_dir, "fold_{}.h5ad".format(job_index)))
