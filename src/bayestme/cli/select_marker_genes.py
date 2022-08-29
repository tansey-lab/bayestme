import argparse

from bayestme import data, deconvolution


def get_parser():
    parser = argparse.ArgumentParser(description='Plot deconvolution results')
    parser.add_argument('--stdata', type=str,
                        help='Input file, SpatialExpressionDataset in h5 format')
    parser.add_argument('--deconvolution-result', type=str,
                        help='Input file, DeconvolutionResult in h5 format')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory.')
    parser.add_argument('--n-marker-genes', type=int,
                        default=5,
                        help='Maximum number of marker genes per cell type.')
    parser.add_argument('--alpha', type=float,
                        default=0.05,
                        help='Alpha cutoff for choosing marker genes.')
    parser.add_argument('--marker-gene-method',
                        type=deconvolution.MarkerGeneMethod,
                        choices=list(deconvolution.MarkerGeneMethod),
                        default=deconvolution.MarkerGeneMethod.TIGHT,
                        help='Method for choosing marker genes.')
    return parser


def main():
    args = get_parser().parse_args()

    stdata = data.SpatialExpressionDataset.read_h5(args.stdata)
    deconvolution_result = data.DeconvolutionResult.read_h5(args.deconvolution_result)

    marker_genes = deconvolution.select_marker_genes(
        deconvolution_result=deconvolution_result,
        n_marker=args.n_marker_genes,
        alpha=args.alpha,
        method=args.marker_gene_method)

    deconvolution.add_marker_gene_results_to_dataset(
        stdata=stdata,
        marker_genes=marker_genes)

    stdata.save(args.stdata)

