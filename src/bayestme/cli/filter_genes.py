import argparse
import logging

import bayestme.log_config
from bayestme import data, gene_filtering

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Filter genes from dataset based on one or more criteria"
    )
    parser.add_argument("--adata", type=str, help="Input AnnData in h5 format")
    parser.add_argument(
        "--output",
        type=str,
        help="Output file, AnnData in h5 format containing the bleed corrected counts",
    )
    parser.add_argument(
        "--filter-ribosomal-genes",
        action="store_true",
        default=False,
        help="Filter ribosomal genes (based on gene name regex)",
    )
    parser.add_argument(
        "--n-top-by-standard-deviation",
        type=int,
        default=None,
        help="Use the top N genes with the highest spatial variance.",
    )
    parser.add_argument(
        "--spot-threshold",
        type=float,
        default=None,
        help="Filter genes appearing in greater than the provided threshold of tissue spots.",
    )
    parser.add_argument(
        "--expression-truth",
        help="Filter out genes not found in all expression truth datasets.",
        type=str,
        action="append",
        default=None,
    )
    bayestme.log_config.add_logging_args(parser)

    return parser


def main():
    args = get_parser().parse_args()
    bayestme.log_config.configure_logging(args)

    dataset = data.SpatialExpressionDataset.read_h5(args.adata)

    if args.n_top_by_standard_deviation is not None:
        logger.info(
            "Will filter the top {} genes by standard deviation.".format(
                args.n_top_by_standard_deviation
            )
        )

        pre_filtering_genes = dataset.gene_names

        dataset = gene_filtering.select_top_genes_by_standard_deviation(
            dataset, n_gene=args.n_top_by_standard_deviation
        )

        post_filtering_genes = dataset.gene_names

        logger.info(
            "After standard deviation filtering went from {} to {} genes. Filtered genes: {}".format(
                len(pre_filtering_genes),
                len(post_filtering_genes),
                ", ".join(set(pre_filtering_genes) - set(post_filtering_genes)),
            )
        )

    if args.spot_threshold is not None:
        pre_filtering_genes = dataset.gene_names

        dataset = gene_filtering.filter_genes_by_spot_threshold(
            dataset, spot_threshold=args.spot_threshold
        )

        post_filtering_genes = dataset.gene_names

        logger.info(
            "After spot_threshold filtering went from {} to {} genes. Filtered genes: {}".format(
                len(pre_filtering_genes),
                len(post_filtering_genes),
                ", ".join(set(pre_filtering_genes) - set(post_filtering_genes)),
            )
        )

    if args.filter_ribosomal_genes:
        pre_filtering_genes = dataset.gene_names

        dataset = gene_filtering.filter_ribosome_genes(dataset)

        post_filtering_genes = dataset.gene_names

        logger.info(
            "After ribosomal filtering went from {} to {} genes. Filtered genes: {}".format(
                len(pre_filtering_genes),
                len(post_filtering_genes),
                ", ".join(set(pre_filtering_genes) - set(post_filtering_genes)),
            )
        )

    if args.expression_truth:
        for expression_truth_file in args.expression_truth:
            pre_filtering_genes = dataset.gene_names

            dataset = gene_filtering.filter_stdata_to_match_expression_truth(
                dataset, expression_truth_file
            )

            post_filtering_genes = dataset.gene_names

            logger.info(
                "After intersecting with expression truth gene set from file {} went from {} to {} genes. Filtered genes: {}".format(
                    expression_truth_file,
                    len(pre_filtering_genes),
                    len(post_filtering_genes),
                    ", ".join(set(pre_filtering_genes) - set(post_filtering_genes)),
                )
            )

    dataset.save(args.output)
