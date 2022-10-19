import argparse

from bayestme import data, bleeding_correction
import bayestme.logging

def get_parser():
    parser = argparse.ArgumentParser(description='Perform bleeding correction')
    parser.add_argument('--adata', type=str,
                        help='Input file, AnnData in h5 format')
    parser.add_argument('--bleed-out', type=str,
                        help='Output file, BleedCorrectionResult in h5 format')
    parser.add_argument('--adata-output', type=str,
                        help='A new AnnData in h5 format created using the bleed corrected counts')
    parser.add_argument('-i', '--inplace', default=False, action='store_true',
                        help='If provided, overwrite the input file --adata')
    parser.add_argument('--n-top', type=int, default=50,
                        help='Use N top genes by standard deviation to calculate the bleeding functions. '
                             'Genes will not be filtered from output dataset.')
    parser.add_argument('--max-steps', type=int, default=5,
                        help='Number of EM steps')
    parser.add_argument('--local-weight', type=int, default=None,
                        help='Initial value for local weight, a tuning parameter for bleed correction. '
                             'rho_0g from equation 1 in the paper. By default will be set to sqrt(N tissue spots)')
    bayestme.logging.add_logging_args(parser)
    return parser


def main():
    args = get_parser().parse_args()

    bayestme.logging.configure_logging(args)

    dataset = data.SpatialExpressionDataset.read_h5(args.adata)

    (cleaned_dataset, bleed_correction_result) = bleeding_correction.clean_bleed(
        dataset=dataset,
        n_top=args.n_top,
        max_steps=args.max_steps,
        local_weight=args.local_weight
    )

    bleed_correction_result.save(args.bleed_out)

    if not args.inplace:
        cleaned_dataset.save(args.adata_output)
    else:
        cleaned_dataset.save(args.adata)
