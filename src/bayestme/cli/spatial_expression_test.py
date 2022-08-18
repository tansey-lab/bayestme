import shutil

import numpy as np
import tempfile
import os
from unittest import mock

from bayestme import data
from bayestme.data_test import generate_toy_stdataset

from bayestme.cli import spatial_expression


def test_spatial_expression():
    dataset = generate_toy_stdataset()

    deconv_result = data.DeconvolutionResult(
        cell_prob_trace=np.zeros((2, 2)),
        expression_trace=np.zeros((2, 2)),
        beta_trace=np.zeros((2, 2)),
        cell_num_trace=np.zeros((2, 2)),
        reads_trace=np.zeros((2, 2)),
        lam2=1000,
        n_components=3
    )

    tmpdir = tempfile.mkdtemp()

    input_path = os.path.join(tmpdir, 'data.h5')
    deconvolve_path = os.path.join(tmpdir, 'deconvolve.h5')
    output = os.path.join(tmpdir, 'result.h5')

    command_line_arguments = [
        'spatial_expression',
        '--deconvolve-results',
        deconvolve_path,
        '--dataset',
        input_path,
        '--output',
        output,
        '--n-spatial-patterns',
        '10',
        '--n-samples',
        '20',
        '--n-thin',
        '2',
        '--n-burn',
        '100',
        '--lam2',
        '1000',
        '--simple'
    ]

    try:
        dataset.save(input_path)
        deconv_result.save(deconvolve_path)

        with mock.patch('sys.argv', command_line_arguments):
            with mock.patch('bayestme.spatial_expression.run_spatial_expression') as run_spatial_expression:
                run_spatial_expression.return_value = data.SpatialDifferentialExpressionResult(
                    w_samples=np.zeros((2, 2)),
                    c_samples=np.zeros((2, 2)),
                    gamma_samples=np.zeros((2, 2)),
                    h_samples=np.zeros((2, 2)),
                    v_samples=np.zeros((2, 2)),
                    theta_samples=np.zeros((2, 2))
                )
                spatial_expression.main()

                run_spatial_expression.assert_called_once_with(
                    sde=mock.ANY,
                    deconvolve_results=mock.ANY,
                    n_samples=20,
                    n_burn=100,
                    n_thin=2,
                    n_cell_min=5,
                    simple=True
                )

                result = data.SpatialDifferentialExpressionResult.read_h5(output)
    finally:
        shutil.rmtree(tmpdir)


def test_spatial_expression_save_state_on_error():
    dataset = generate_toy_stdataset()

    deconv_result = data.DeconvolutionResult(
        cell_prob_trace=np.zeros((2, 2)),
        expression_trace=np.zeros((2, 2)),
        beta_trace=np.zeros((2, 2)),
        cell_num_trace=np.zeros((2, 2)),
        reads_trace=np.zeros((2, 2)),
        lam2=1000,
        n_components=3
    )

    tmpdir = tempfile.mkdtemp()

    input_path = os.path.join(tmpdir, 'data.h5')
    deconvolve_path = os.path.join(tmpdir, 'deconvolve.h5')
    output = os.path.join(tmpdir, 'result.h5')

    command_line_arguments = [
        'spatial_expression',
        '--deconvolve-results',
        deconvolve_path,
        '--dataset',
        input_path,
        '--output',
        output,
        '--n-spatial-patterns',
        '10',
        '--n-samples',
        '20',
        '--n-thin',
        '2',
        '--n-burn',
        '100',
        '--lam2',
        '1000',
        '--simple'
    ]

    try:
        dataset.save(input_path)
        deconv_result.save(deconvolve_path)

        with mock.patch('sys.argv', command_line_arguments):
            with mock.patch('bayestme.spatial_expression.run_spatial_expression') as run_spatial_expression:
                run_spatial_expression.side_effect = ZeroDivisionError()
                try:
                    spatial_expression.main()
                except ZeroDivisionError:
                    pass
                run_spatial_expression.assert_called_once_with(
                    sde=mock.ANY,
                    deconvolve_results=mock.ANY,
                    n_samples=20,
                    n_burn=100,
                    n_thin=2,
                    n_cell_min=5,
                    simple=True
                )

                result = data.SpatialDifferentialExpressionSamplerState.read_h5(
                    os.path.join(os.path.dirname(output), spatial_expression.MODEL_DUMP_PATH))
    finally:
        shutil.rmtree(tmpdir)
