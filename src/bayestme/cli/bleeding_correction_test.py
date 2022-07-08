import shutil

import numpy as np
import tempfile
import os

from unittest import mock
from bayestme import data
from bayestme.cli import bleeding_correction
from bayestme.data_test import generate_toy_stdataset


def test_filter_genes():
    dataset = generate_toy_stdataset()
    cleaning_results = data.BleedCorrectionResult(
        corrected_reads=np.zeros((2, 2)),
        global_rates=np.zeros((2, 2)),
        basis_functions=np.zeros((2, 2)),
        weights=np.zeros((2, 2))
    )

    tmpdir = tempfile.mkdtemp()

    input_path = os.path.join(tmpdir, 'data.h5')
    bleed_out = os.path.join(tmpdir, 'bleed.h5')
    clean_out = os.path.join(tmpdir, 'cleaned.h5')

    command_line_arguments = [
        'bleeding_correction',
        '--input',
        input_path,
        '--bleed-out',
        bleed_out,
        '--stdata-out',
        clean_out,
        '--n-top',
        '3',
        '--max-steps',
        '5',
        '--local-weight',
        '15']

    try:
        dataset.save(os.path.join(tmpdir, 'data.h5'))

        with mock.patch('sys.argv', command_line_arguments):
            with mock.patch('bayestme.bleeding_correction.clean_bleed') as clean_bleed:
                clean_bleed.return_value = (
                    dataset, cleaning_results
                )

                bleeding_correction.main()

                data.BleedCorrectionResult.read_h5(bleed_out)
                data.SpatialExpressionDataset.read_h5(clean_out)

                clean_bleed.assert_called_once_with(
                    dataset=mock.ANY,
                    n_top=3,
                    max_steps=5,
                    local_weight=15
                )
    finally:
        shutil.rmtree(tmpdir)
