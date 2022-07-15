import argparse

from bayestme import data, deconvolution

parser = argparse.ArgumentParser(description='Plot deconvolution results')
parser.add_argument('--stdata', type=str,
                    help='Input file, SpatialExpressionDataset in h5 format')
parser.add_argument('--deconvolution-result', type=str,
                    help='Input file, DeconvolutionResult in h5 format')
parser.add_argument('--output-dir', type=str,
                    help='Output directory.')
parser.add_argument('--n-marker-genes', type=int,
                    default=5,
                    help='Plot top N marker genes.')
parser.add_argument('--alpha', type=float,
                    default=0.05,
                    help='Alpha cutoff for choosing marker genes.')
parser.add_argument('--marker-gene-method',
                    type=deconvolution.MarkerGeneMethod,
                    choices=list(deconvolution.MarkerGeneMethod),
                    default=deconvolution.MarkerGeneMethod.TIGHT,
                    help='Method for choosing marker genes.')


def main():
    args = parser.parse_args()

    stdata = data.SpatialExpressionDataset.read_h5(args.stdata)
    deconvolution_result = data.DeconvolutionResult.read_h5(args.deconvolution_result)

    deconvolution.plot_deconvolution(
        stdata=stdata,
        deconvolution_result=deconvolution_result,
        output_dir=args.output_dir,
        n_marker_genes=args.n_marker_genes,
        alpha=args.alpha,
        marker_gene_method=args.marker_gene_method)
