import tempfile
import os.path
import numpy as np
import shutil
import pathlib

from bayestme import cv_likelihoods


def create_fake_data(n_components, lam_vals, n_fold, n_draw, n_gene=1000, max_ncell=120):
    tempdir = tempfile.mkdtemp()

    for cohort in ['train', 'test']:
        for n_components in range(2, n_components + 1):
            for lam in lam_vals:
                for iter in range(0, n_fold):
                    fn = os.path.join(
                        tempdir,
                        f'exp_{n_gene}_{max_ncell}_{cohort}_likelihood_{n_components}_{lam:.1f}_{iter}.npy')
                    np.save(fn, np.random.random((n_draw,)))
    return tempdir


def test_load_likelihoods():
    lam_vals = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    n_fold = 5
    n_components = 12
    tempdir = create_fake_data(
        lam_vals=lam_vals,
        n_fold=n_fold,
        n_components=n_components,
        n_draw=100
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
        n_draw=100
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
        n_draw=100
    )

    try:
        pathlib.Path(os.path.join(tempdir, 'plot')).mkdir(parents=True, exist_ok=True)
        cv_likelihoods.plot_cv_running(tempdir, os.path.join(tempdir, 'plot'))

        assert os.path.exists(os.path.join(tempdir, 'plot/cv_running.pdf'))
        assert os.path.exists(os.path.join(tempdir, 'plot/k-folds.pdf'))
    finally:
        shutil.rmtree(tempdir)


def test_get_max_likelihood_n_components():
    lam_vals = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    n_fold = 5
    n_components = 12
    likelihoods = np.random.random((2, n_components - 1, len(lam_vals), n_fold))
    n_components = cv_likelihoods.get_max_likelihood_n_components(likelihoods)

    assert n_components in range(2, n_components + 1)
