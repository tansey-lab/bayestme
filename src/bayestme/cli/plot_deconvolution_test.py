import numpy
import numpy as np
import os
import tempfile
import bayestme.cli.plot_deconvolution

from unittest import mock

import bayestme.synthetic_data
from bayestme import data, deconvolution, bleeding_correction, deconvolution_test


def test_plot_deconvolution():
    np.random.seed(101)
    n_genes = 50
    n_marker_genes = 5
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12,
        n_cols=12,
        n_genes=n_genes)

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(['gene{}'.format(x) for x in range(n_genes)]),
        layout=data.Layout.SQUARE)

    deconvolve_results = deconvolution_test.create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=5,
        n_samples=100,
        n_gene=dataset.n_gene)

    tmpdir = tempfile.mkdtemp()

    stdata_fn = os.path.join(tmpdir, 'data.h5')
    deconvolve_results_fn = os.path.join(tmpdir, 'deconvolve.h5')
    dataset.save(stdata_fn)
    deconvolve_results.save(deconvolve_results_fn)

    command_line_args = [
        'plot_deconvolution',
        '--stdata',
        stdata_fn,
        '--output-dir',
        tmpdir]

    with mock.patch('bayestme.deconvolution.plot_deconvolution') as plot_deconvolution_mock:
        with mock.patch('sys.argv', command_line_args):
            bayestme.cli.plot_deconvolution.main()

            plot_deconvolution_mock.assert_called_once_with(
                stdata=mock.ANY,
                output_dir=tmpdir,
                cell_type_names=None)


def test_plot_deconvolution_with_cell_type_names():
    np.random.seed(101)
    n_genes = 50
    n_marker_genes = 5
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12,
        n_cols=12,
        n_genes=n_genes)

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(['gene{}'.format(x) for x in range(n_genes)]),
        layout=data.Layout.SQUARE)

    deconvolve_results = deconvolution_test.create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=5,
        n_samples=100,
        n_gene=dataset.n_gene)

    tmpdir = tempfile.mkdtemp()

    stdata_fn = os.path.join(tmpdir, 'data.h5')
    deconvolve_results_fn = os.path.join(tmpdir, 'deconvolve.h5')
    dataset.save(stdata_fn)
    deconvolve_results.save(deconvolve_results_fn)

    command_line_args = [
        'plot_deconvolution',
        '--stdata',
        stdata_fn,
        '--output-dir',
        tmpdir,
        '--cell-type-names',
        'type 1, type 2, type 3']

    with mock.patch('bayestme.deconvolution.plot_deconvolution') as plot_deconvolution_mock:
        with mock.patch('sys.argv', command_line_args):
            bayestme.cli.plot_deconvolution.main()

            plot_deconvolution_mock.assert_called_once_with(
                stdata=mock.ANY,
                output_dir=tmpdir,
                cell_type_names=['type 1', 'type 2', 'type 3'])
