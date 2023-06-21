import os
import tempfile
from unittest import mock

import numpy as np

from bayestme.cli import phenotype_selection
from bayestme.common import InferenceType
from bayestme.data_test import generate_toy_stdataset


def test_phenotype_selection_all_job():
    dataset = generate_toy_stdataset()
    tmpdir = tempfile.mkdtemp()

    stdata_fn = os.path.join(tmpdir, "data.h5")
    dataset.save(stdata_fn)

    command_line_args = [
        "phenotype_selection",
        "--adata",
        stdata_fn,
        "--n-fold",
        "1",
        "--n-splits",
        "15",
        "--n-samples",
        "100",
        "--n-thin",
        "3",
        "--n-burn",
        "1000",
        "--spatial-smoothing-values",
        "1",
        "--n-components-min",
        "4",
        "--n-components-max",
        "6",
        "--output-dir",
        tmpdir,
    ]

    with mock.patch(
        "bayestme.phenotype_selection.run_phenotype_selection_single_job"
    ) as run_phenotype_selection_single_job_mock:
        with mock.patch(
            "bayestme.phenotype_selection.get_phenotype_selection_parameters_for_folds"
        ) as get_phenotype_selection_parameters_for_folds_mock:
            with mock.patch("sys.argv", command_line_args):
                get_phenotype_selection_parameters_for_folds_mock.return_value = [
                    (1, 4, np.ones(25), 0),
                    (1, 5, np.ones(25), 0),
                    (1, 6, np.ones(25), 0),
                ]

                phenotype_selection.main()

                run_phenotype_selection_single_job_mock.assert_has_calls(
                    [
                        mock.call(
                            stdata=mock.ANY,
                            spatial_smoothing_parameter=1,
                            n_components=4,
                            mask=mock.ANY,
                            fold_number=0,
                            n_samples=100,
                            mcmc_n_burn=1000,
                            mcmc_n_thin=3,
                            n_svi_steps=10_000,
                            background_noise=False,
                            lda_initialization=False,
                            inference_type=InferenceType.MCMC,
                            rng=mock.ANY,
                        ),
                        mock.call(
                            stdata=mock.ANY,
                            spatial_smoothing_parameter=1,
                            n_components=5,
                            mask=mock.ANY,
                            fold_number=0,
                            n_samples=100,
                            mcmc_n_burn=1000,
                            mcmc_n_thin=3,
                            n_svi_steps=10_000,
                            background_noise=False,
                            lda_initialization=False,
                            inference_type=InferenceType.MCMC,
                            rng=mock.ANY,
                        ),
                        mock.call(
                            stdata=mock.ANY,
                            spatial_smoothing_parameter=1,
                            n_components=6,
                            mask=mock.ANY,
                            fold_number=0,
                            n_samples=100,
                            mcmc_n_burn=1000,
                            mcmc_n_thin=3,
                            n_svi_steps=10_000,
                            background_noise=False,
                            lda_initialization=False,
                            inference_type=InferenceType.MCMC,
                            rng=mock.ANY,
                        ),
                    ],
                    any_order=True,
                )


def test_phenotype_selection_single_job():
    dataset = generate_toy_stdataset()
    tmpdir = tempfile.mkdtemp()

    stdata_fn = os.path.join(tmpdir, "data.h5")
    dataset.save(stdata_fn)

    command_line_args = [
        "phenotype_selection",
        "--adata",
        stdata_fn,
        "--job-index",
        "0",
        "--n-fold",
        "1",
        "--n-splits",
        "15",
        "--n-samples",
        "100",
        "--n-thin",
        "3",
        "--n-burn",
        "1000",
        "--spatial-smoothing-values",
        "1",
        "--n-components-min",
        "4",
        "--n-components-max",
        "4",
        "--output-dir",
        tmpdir,
    ]

    with mock.patch(
        "bayestme.phenotype_selection.run_phenotype_selection_single_job"
    ) as run_phenotype_selection_single_job_mock:
        with mock.patch(
            "bayestme.phenotype_selection.get_phenotype_selection_parameters_for_folds"
        ) as get_phenotype_selection_parameters_for_folds_mock:
            with mock.patch("sys.argv", command_line_args):
                get_phenotype_selection_parameters_for_folds_mock.return_value = [
                    (1, 4, np.ones(25), 0)
                ]

                phenotype_selection.main()

                get_phenotype_selection_parameters_for_folds_mock.assert_called_once_with(
                    stdata=mock.ANY,
                    n_fold=1,
                    n_splits=15,
                    lams=[1],
                    n_components_min=4,
                    n_components_max=4,
                )

                run_phenotype_selection_single_job_mock.assert_called_once_with(
                    stdata=mock.ANY,
                    spatial_smoothing_parameter=1,
                    n_components=4,
                    mask=mock.ANY,
                    fold_number=0,
                    n_samples=100,
                    mcmc_n_burn=1000,
                    mcmc_n_thin=3,
                    n_svi_steps=10_000,
                    background_noise=False,
                    lda_initialization=False,
                    inference_type=InferenceType.MCMC,
                    rng=mock.ANY,
                )
