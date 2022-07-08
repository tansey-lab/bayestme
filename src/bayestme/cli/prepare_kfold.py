import argparse
from bayestme import bayestme_data

parser = argparse.ArgumentParser(description='Deconvolve data')
parser.add_argument('--data-dir', type=str,
                    help='input data dir')
parser.add_argument('--n-fold', type=int, default=5)
parser.add_argument('--n-splits', type=int, default=15)
parser.add_argument('--n-samples', type=int, default=100)
parser.add_argument('--n-burn', type=int, default=2000)
parser.add_argument('--n-thin', type=int, default=5)
parser.add_argument('--lda', type=int, default=0)
parser.add_argument('--n-comp-min', type=int, default=2)
parser.add_argument('--n-comp-max', type=int, default=12)
parser.add_argument('--lambda-values', type=float, action='append', default=(1, 1e1, 1e2, 1e3, 1e4, 1e5))
parser.add_argument('--max-ncell', type=int, default=120)


def main():
    args = parser.parse_args()

    try:
        stdata = bayestme_data.CleanedSTData(load_path=args.data_dir)
        stdata.load_data('BayesTME')
    except FileNotFoundError:
        stdata = bayestme_data.RawSTData('BayesTME', load=args.data_dir, storage_path=args.data_dir)
        stdata.load(args.data_dir, storage_path=args.data_dir)

    cv_stdata = bayestme_data.CrossValidationSTData(stdata=stdata,
                                                    n_fold=args.n_fold,
                                                    n_splits=args.n_splits,
                                                    n_samples=args.n_samples,
                                                    n_burn=args.n_burn,
                                                    n_thin=args.n_thin,
                                                    lda=args.lda,
                                                    n_comp_min=args.n_comp_min,
                                                    n_comp_max=args.n_comp_max,
                                                    lambda_values=args.lambda_values,
                                                    max_ncell=args.max_ncell)

    cv_stdata.prepare_jobs()
