import argparse
import logging

import bayestme.log_config
from bayestme import cv_likelihoods

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Read phenotype selection results and print to stdout."
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        help="Dir to write plot to",
    )
    parser.add_argument(
        "--phenotype-selection-output-dir",
        type=str,
        default=None,
        help="Directory with phenotype selection outputs",
    )
    parser.add_argument(
        "--phenotype-selection-outputs",
        type=str,
        nargs="+",
        help="Individual phenotype selection outputs",
    )
    parser.add_argument(
        "--spatial-smoothing-values",
        type=float,
        action="append",
        help="Potential values of the spatial smoothing parameter to try. "
        "Defaults to (1, 1e1, 1e2, 1e3, 1e4, 1e5)",
    )
    parser.add_argument(
        "--output-n-components",
        type=str,
        help="Output file to write results to (for use in scripting and pipelines).",
    )
    parser.add_argument(
        "--output-lambda",
        type=str,
        help="Output file to write results to (for use in scripting and pipelines).",
    )
    bayestme.log_config.add_logging_args(parser)

    return parser


def main():
    args = get_parser().parse_args()
    bayestme.log_config.configure_logging(args)

    likelihoods, fold_nums, lam_vals, k_vals = cv_likelihoods.load_likelihoods(
        output_dir=args.phenotype_selection_output_dir,
        output_files=args.phenotype_selection_outputs,
    )

    max_likelihood_n_components = cv_likelihoods.get_max_likelihood_n_components(
        likelihoods, k_vals
    )

    max_likelihood_lambda_value = cv_likelihoods.get_best_lambda_value(
        likelihoods, max_likelihood_n_components, args.spatial_smoothing_values, k_vals
    )

    logger.info(
        f"Max likelihood n_components: {max_likelihood_n_components} lambda: {max_likelihood_lambda_value}"
    )

    cv_likelihoods.plot_cv_running(
        args.phenotype_selection_output_dir,
        args.phenotype_selection_outputs,
        out_path=args.plot_output,
    )

    with open(args.output_n_components, "w") as of:
        of.write("{}".format(max_likelihood_n_components))

    with open(args.output_lambda, "w") as of:
        of.write("{}".format(max_likelihood_lambda_value))
