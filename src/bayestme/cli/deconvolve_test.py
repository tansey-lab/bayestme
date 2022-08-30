import shutil

import numpy as np
import tempfile
import os

from unittest import mock
from bayestme import data, deconvolution_test
from bayestme.cli import deconvolve
from bayestme.data_test import generate_toy_stdataset


def test_deconvolve():
    dataset = generate_toy_stdataset()
    tmpdir = tempfile.mkdtemp()

    input_path = os.path.join(tmpdir, 'data.h5')
    output_path = os.path.join(tmpdir, 'deconvolve.h5')

    deconvolve_rv = deconvolution_test.create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=5,
        n_samples=100,
        n_gene=dataset.n_gene
    )

    command_line_arguments = [
        'deconvolve',
        '--input', input_path,
        '--output', output_path,
        '--n-gene', '1000',
        '--lam2', '1000',
        '--n-samples', '100',
        '--n-burnin', '500',
        '--n-thin', '2',
        '--n-components', '5'
    ]

    try:
        dataset.save(input_path)

        with mock.patch('sys.argv', command_line_arguments):
            with mock.patch('bayestme.deconvolution.deconvolve') as deconvolve_mock:
                deconvolve_mock.return_value = deconvolve_rv

                deconvolve.main()

                data.DeconvolutionResult.read_h5(output_path)

                deconvolve_mock.assert_called_once_with(
                    reads=mock.ANY,
                    edges=mock.ANY,
                    n_gene=1000,
                    n_components=5,
                    lam2=1000,
                    n_samples=100,
                    n_burnin=500,
                    n_thin=2,
                    bkg=False,
                    lda=False,
                    expression_truth=None,
                    rng=mock.ANY
                )
    finally:
        shutil.rmtree(tmpdir)


def test_deconvolve_with_expression_truth():
    dataset = generate_toy_stdataset()
    tmpdir = tempfile.mkdtemp()

    input_path = os.path.join(tmpdir, 'data.h5')
    output_path = os.path.join(tmpdir, 'deconvolve.h5')

    deconvolve_rv = deconvolution_test.create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=5,
        n_samples=100,
        n_gene=dataset.n_gene
    )

    command_line_arguments = [
        'deconvolve',
        '--input', input_path,
        '--output', output_path,
        '--n-gene', '1000',
        '--lam2', '1000',
        '--n-samples', '100',
        '--n-burnin', '500',
        '--n-thin', '2',
        '--expression-truth', 'xxx'
    ]

    try:
        dataset.save(input_path)

        with mock.patch('sys.argv', command_line_arguments):
            with mock.patch('bayestme.deconvolution.deconvolve') as deconvolve_mock:
                with mock.patch('bayestme.deconvolution.load_expression_truth') as load_expression_truth_mock:
                    expression_truth = np.zeros((9, 10))
                    load_expression_truth_mock.return_value = expression_truth
                    deconvolve_mock.return_value = deconvolve_rv

                    deconvolve.main()

                    data.DeconvolutionResult.read_h5(output_path)

                    deconvolve_mock.assert_called_once_with(
                        reads=mock.ANY,
                        edges=mock.ANY,
                        n_gene=1000,
                        n_components=9,
                        lam2=1000,
                        n_samples=100,
                        n_burnin=500,
                        n_thin=2,
                        bkg=False,
                        lda=False,
                        expression_truth=mock.ANY,
                        rng=mock.ANY
                    )
    finally:
        shutil.rmtree(tmpdir)
