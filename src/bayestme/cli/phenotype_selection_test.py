import numpy
import numpy as np
import os
import tempfile

from unittest import mock
from bayestme.cli import phenotype_selection

from bayestme.data_test import generate_toy_stdataset


def test_phenotype_selection():
    np.random.seed(101)

    dataset = generate_toy_stdataset()
    tmpdir = tempfile.mkdtemp()

    stdata_fn = os.path.join(tmpdir, 'data.h5')
    dataset.save(stdata_fn)

    command_line_args = [
        'phenotype_selection',
        '--stdata',
        stdata_fn,
        '--fold-idx',
        '0',
        '--n-fold',
        '1',
        '--n-splits',
        '15',
        '--n-samples',
        '100',
        '--n-thin',
        '3',
        '--n-burn',
        '1000',
        '--n-gene',
        '50',
        '--lambda-values',
        '1',
        '--lambda-values',
        '10',
        '--n-components-min',
        '4',
        '--n-components-max',
        '20',
        '--max-ncell',
        '99',
        '--n-gene',
        '999',
        '--output-dir',
        tmpdir]

    with mock.patch(
            'bayestme.phenotype_selection.run_phenotype_selection_single_fold') as run_phenotype_selection_single_fold_mock:
        with mock.patch('sys.argv', command_line_args):
            phenotype_selection.main()

            run_phenotype_selection_single_fold_mock.assert_called_once_with(
                fold_idx=0,
                stdata=mock.ANY,
                n_fold=1,
                n_splits=15,
                lams=[1, 10],
                n_components_min=4,
                n_components_max=20,
                n_samples=100,
                n_burn=1000,
                n_thin=3,
                max_ncell=99,
                n_gene=999,
                background_noise=False,
                lda_initialization=False)
