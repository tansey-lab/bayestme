import numpy
import numpy as np
import os
import tempfile

from unittest import mock
from bayestme import data
from bayestme.cli import plot_bleeding_correction

from bayestme.data_test import generate_toy_stdataset


def test_plot_bleeding_correction():
    np.random.seed(101)

    dataset = generate_toy_stdataset()
    cleaning_results = data.BleedCorrectionResult(
        corrected_reads=numpy.zeros((2, 2)),
        global_rates=numpy.zeros((2, 2)),
        basis_functions=numpy.zeros((2, 2)),
        weights=numpy.zeros((2, 2))
    )

    tmpdir = tempfile.mkdtemp()

    raw_out = os.path.join(tmpdir, 'data.h5')
    bleed_out = os.path.join(tmpdir, 'bleed.h5')
    clean_out = os.path.join(tmpdir, 'cleaned.h5')
    dataset.save(raw_out)
    dataset.save(clean_out)
    cleaning_results.save(bleed_out)

    command_line_args = [
        'plot_bleeding_correction',
        '--raw-stdata',
        raw_out,
        '--corrected-stdata',
        clean_out,
        '--bleed-correction-results',
        bleed_out,
        '--output-dir',
        tmpdir,
        '--n-top',
        '3']

    with mock.patch(
            'bayestme.bleeding_correction.create_top_n_gene_bleeding_plots') as create_top_n_gene_bleeding_plots:
        with mock.patch('bayestme.bleeding_correction.plot_basis_functions') as plot_basis_functions:
            with mock.patch('sys.argv', command_line_args):
                plot_bleeding_correction.main()

                create_top_n_gene_bleeding_plots.assert_called_once_with(
                    dataset=mock.ANY,
                    corrected_dataset=mock.ANY,
                    bleed_result=mock.ANY,
                    output_dir=tmpdir,
                    n_genes=3
                )
                plot_basis_functions.assert_called_once_with(
                    basis_functions=mock.ANY,
                    output_dir=tmpdir
                )
