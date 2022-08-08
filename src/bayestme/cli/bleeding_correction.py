import argparse

from bayestme import data, bleeding_correction


def get_parser():
    parser = argparse.ArgumentParser(description='Filter data')
    parser.add_argument('--input', type=str,
                        help='Input file, SpatialExpressionDataset in h5 format')
    parser.add_argument('--bleed-out', type=str,
                        help='Output file, BleedCorrectionResult in h5 format')
    parser.add_argument('--stdata-out', type=str,
                        help='Output file, SpatialExpressionDataset in h5 format')
    parser.add_argument('--n-top', type=int, default=50,
                        help='Use N top genes by standard deviation to calculate the bleeding functions. '
                             'Genes will not be filtered from output dataset.')
    parser.add_argument('--max-steps', type=int, default=5,
                        help='Number of EM steps')
    parser.add_argument('--local-weight', type=int, default=None,
                        help='Initial value for local weight, a tuning parameter for bleed correction. '
                             'rho_0g from equation 1 in the paper. By default will be set to sqrt(N tissue spots)')

    return parser


def main():
    args = get_parser().parse_args()

    dataset = data.SpatialExpressionDataset.read_h5(args.input)

    (cleaned_dataset, bleed_correction_result) = bleeding_correction.clean_bleed(
        dataset=dataset,
        n_top=args.n_top,
        max_steps=args.max_steps,
        local_weight=args.local_weight
    )

    bleed_correction_result.save(args.bleed_out)
    cleaned_dataset.save(args.stdata_out)
