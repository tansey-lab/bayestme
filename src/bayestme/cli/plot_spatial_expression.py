import argparse

from bayestme import data, spatial_expression

parser = argparse.ArgumentParser(description='Plot deconvolution results')
parser.add_argument('--stdata', type=str,
                    help='Input file, SpatialExpressionDataset in h5 format')
parser.add_argument('--deconvolution-result', type=str,
                    help='Input file, DeconvolutionResult in h5 format')
parser.add_argument('--sde-result', type=str,
                    help='Input file, SpatialDifferentialExpressionResult in h5 format')
parser.add_argument('--output-dir', type=str,
                    help='Output directory')


def main():
    args = parser.parse_args()

    stdata = data.SpatialExpressionDataset.read_h5(args.stdata)
    deconvolution_result = data.DeconvolutionResult.read_h5(args.deconvolution_result)
    sde_result = data.SpatialDifferentialExpressionResult.read_h5(args.sde_result)

    spatial_expression.plot_significant_spatial_patterns(
        stdata=stdata,
        decon_result=deconvolution_result,
        sde_result=sde_result,
        output_dir=args.output_dir)
