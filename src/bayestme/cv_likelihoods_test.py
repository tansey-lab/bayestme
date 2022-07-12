import tempfile
import os.path
import numpy as np
import shutil
import pathlib

from bayestme import cv_likelihoods, data


def create_fake_data(n_components,
                     lam_vals,
                     n_fold,
                     n_samples,
                     n_spot_in,
                     n_gene=10):
    tempdir = tempfile.mkdtemp()
    i = 0

    for cohort in ['train', 'test']:
        for n_components in range(2, n_components + 1):
            for lam in lam_vals:
                for fold_num in range(0, n_fold):
                    fn = os.path.join(
                        tempdir,
                        f'fold_{i}.h5ad')

                    cell_prob_trace = np.random.random((n_samples, n_spot_in, n_components + 1))
                    cell_num_trace = np.random.random((n_samples, n_spot_in, n_components + 1))
                    expression_trace = np.random.random((n_samples, n_components, n_gene))
                    beta_trace = np.random.random((n_samples, n_components))
                    loglhtest_trace = np.random.random(n_samples)
                    loglhtrain_trace = np.random.random(n_samples)

                    result = data.PhenotypeSelectionResult(
                        cell_num_trace=cell_num_trace,
                        cell_prob_trace=cell_prob_trace,
                        expression_trace=expression_trace,
                        beta_trace=beta_trace,
                        log_lh_train_trace=loglhtrain_trace,
                        log_lh_test_trace=loglhtest_trace,
                        n_components=n_components,
                        lam=lam,
                        fold_number=fold_num,
                        mask=np.ones(n_spot_in)
                    )

                    result.save(fn)

                    i += 1
    return tempdir


def test_load_likelihoods():
    lam_vals = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    n_fold = 5
    n_components = 12
    tempdir = create_fake_data(
        lam_vals=lam_vals,
        n_fold=n_fold,
        n_components=n_components,
        n_samples=100,
        n_spot_in=25
    )

    try:
        likelihoods, fold_nums, lam_vals, k_vals = cv_likelihoods.load_likelihoods(tempdir)

        assert fold_nums == list(range(n_fold))
        assert lam_vals == [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
        assert k_vals == list(range(2, n_components + 1))
        assert likelihoods.shape == (2, n_components - 1, len(lam_vals), n_fold)
    finally:
        shutil.rmtree(tempdir)


def test_plot_likelihoods():
    lam_vals = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    n_fold = 5
    n_components = 12
    tempdir = create_fake_data(
        lam_vals=lam_vals,
        n_fold=n_fold,
        n_components=n_components,
        n_samples=100,
        n_spot_in=25
    )

    try:
        cv_likelihoods.plot_likelihoods(tempdir, os.path.join(tempdir, 'plot'))

        assert os.path.exists(os.path.join(tempdir, 'plot.png'))
    finally:
        shutil.rmtree(tempdir)


def test_plot_cv_running():
    lam_vals = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    n_fold = 5
    n_components = 12
    tempdir = create_fake_data(
        lam_vals=lam_vals,
        n_fold=n_fold,
        n_components=n_components,
        n_samples=100,
        n_spot_in=25
    )

    try:
        pathlib.Path(os.path.join(tempdir, 'plot')).mkdir(parents=True, exist_ok=True)
        cv_likelihoods.plot_cv_running(tempdir, os.path.join(tempdir, 'plot'))

        assert os.path.exists(os.path.join(tempdir, 'plot/cv_running.pdf'))
        assert os.path.exists(os.path.join(tempdir, 'plot/k_folds.pdf'))
    finally:
        shutil.rmtree(tempdir)


def test_get_max_likelihood_n_components():
    lam_vals = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    n_fold = 5
    k_vals = [2, 3, 4, 5, 6, 7, 8]
    likelihoods = np.random.random((2, len(k_vals), len(lam_vals), n_fold))
    n_components = cv_likelihoods.get_max_likelihood_n_components(
        likelihoods,
        k_vals=k_vals)

    best_lambda = cv_likelihoods.get_best_lambda_value(
        likelihoods=likelihoods,
        lambda_array=lam_vals,
        best_n_components=n_components,
        k_vals=k_vals)

    assert n_components in k_vals
    assert best_lambda in lam_vals
