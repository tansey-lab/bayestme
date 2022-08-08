import argparse

from bayestme import data, bleeding_correction


def get_parser():
    parser = argparse.ArgumentParser(description='Plot bleeding correction results')
    parser.add_argument('--raw-stdata', type=str,
                        help='Input file, SpatialExpressionDataset in h5 format')
    parser.add_argument('--corrected-stdata', type=str,
                        help='Input file, SpatialExpressionDataset in h5 format')
    parser.add_argument('--bleed-correction-results', type=str,
                        help='Input file, BleedCorrectionResult in h5 format')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory')
    parser.add_argument('--n-top', type=int,
                        default=10,
                        help='Plot top n genes by stddev')
    return parser


def main():
    args = get_parser().parse_args()

    before_correction = data.SpatialExpressionDataset.read_h5(args.raw_stdata)
    after_correction = data.SpatialExpressionDataset.read_h5(args.corrected_stdata)

    bleeding_correction_results = data.BleedCorrectionResult.read_h5(args.bleed_correction_results)

    bleeding_correction.create_top_n_gene_bleeding_plots(
        dataset=before_correction,
        corrected_dataset=after_correction,
        bleed_result=bleeding_correction_results,
        output_dir=args.output_dir,
        n_genes=args.n_top
    )

    bleeding_correction.plot_basis_functions(
        basis_functions=bleeding_correction_results.basis_functions,
        output_dir=args.output_dir
    )
