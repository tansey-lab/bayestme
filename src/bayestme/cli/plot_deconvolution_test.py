import os
import tempfile
from unittest import mock

import numpy as np
import pandas

import bayestme.cli.plot_deconvolution
import bayestme.common
import bayestme.synthetic_data
import bayestme.utils
from bayestme import data


def test_plot_deconvolution():
    np.random.seed(101)
    n_genes = 50
    n_marker_genes = 5
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12, n_cols=12, n_genes=n_genes
    )

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(["gene{}".format(x) for x in range(n_genes)]),
        layout=bayestme.common.Layout.SQUARE,
        edges=bayestme.utils.get_edges(locations, bayestme.common.Layout.SQUARE),
    )

    deconvolve_results = bayestme.synthetic_data.create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in, n_components=5, n_samples=100, n_gene=dataset.n_gene
    )

    tmpdir = tempfile.mkdtemp()

    stdata_fn = os.path.join(tmpdir, "data.h5")
    deconvolve_results_fn = os.path.join(tmpdir, "deconvolve.h5")
    dataset.save(stdata_fn)
    deconvolve_results.save(deconvolve_results_fn)

    command_line_args = [
        "plot_deconvolution",
        "--adata",
        stdata_fn,
        "--output-dir",
        tmpdir,
    ]

    with mock.patch(
        "bayestme.plot.deconvolution.plot_deconvolution"
    ) as plot_deconvolution_mock:
        with mock.patch("sys.argv", command_line_args):
            bayestme.cli.plot_deconvolution.main()

            plot_deconvolution_mock.assert_called_once_with(
                stdata=mock.ANY, output_dir=tmpdir, cell_type_names=None
            )


def test_plot_deconvolution_with_cell_type_names():
    np.random.seed(101)
    n_genes = 50
    n_marker_genes = 5
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12, n_cols=12, n_genes=n_genes
    )

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(["gene{}".format(x) for x in range(n_genes)]),
        layout=bayestme.common.Layout.SQUARE,
        edges=bayestme.utils.get_edges(locations, bayestme.common.Layout.SQUARE),
    )

    deconvolve_results = bayestme.synthetic_data.create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in, n_components=5, n_samples=100, n_gene=dataset.n_gene
    )

    tmpdir = tempfile.mkdtemp()

    stdata_fn = os.path.join(tmpdir, "data.h5")
    deconvolve_results_fn = os.path.join(tmpdir, "deconvolve.h5")
    dataset.save(stdata_fn)
    deconvolve_results.save(deconvolve_results_fn)

    command_line_args = [
        "plot_deconvolution",
        "--adata",
        stdata_fn,
        "--output-dir",
        tmpdir,
        "--cell-type-names",
        "type 1, type 2, type 3",
    ]

    with mock.patch(
        "bayestme.plot.deconvolution.plot_deconvolution"
    ) as plot_deconvolution_mock:
        with mock.patch("sys.argv", command_line_args):
            bayestme.cli.plot_deconvolution.main()

            plot_deconvolution_mock.assert_called_once_with(
                stdata=mock.ANY,
                output_dir=tmpdir,
                cell_type_names=["type 1", "type 2", "type 3"],
            )


def test_plot_deconvolution_with_cell_type_names_from_exp_truth():
    np.random.seed(101)
    n_genes = 50
    n_marker_genes = 5
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12, n_cols=12, n_genes=n_genes
    )

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(["gene{}".format(x) for x in range(n_genes)]),
        layout=bayestme.common.Layout.SQUARE,
        edges=bayestme.utils.get_edges(locations, bayestme.common.Layout.SQUARE),
    )

    deconvolve_results = bayestme.synthetic_data.create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in, n_components=5, n_samples=100, n_gene=dataset.n_gene
    )

    bayestme.data.add_deconvolution_results_to_dataset(
        stdata=dataset, result=deconvolve_results
    )

    fake_expression_truth = pandas.DataFrame(np.random.poisson(10, size=(50, 5)))
    fake_expression_truth.index = np.array(["gene{}".format(x) for x in range(n_genes)])
    fake_expression_truth.columns = ["type 1", "type 2", "type 3", "type 4", "type 5"]

    tmpdir = tempfile.mkdtemp()

    fake_expression_truth_fn = os.path.join(tmpdir, "expression_truth.csv")
    fake_expression_truth.to_csv(fake_expression_truth_fn)

    stdata_fn = os.path.join(tmpdir, "data.h5")
    deconvolve_results_fn = os.path.join(tmpdir, "deconvolve.h5")
    dataset.save(stdata_fn)
    deconvolve_results.save(deconvolve_results_fn)

    command_line_args = [
        "plot_deconvolution",
        "--adata",
        stdata_fn,
        "--output-dir",
        tmpdir,
        "--expression-truth",
        fake_expression_truth_fn,
    ]

    with mock.patch(
        "bayestme.plot.deconvolution.plot_deconvolution"
    ) as plot_deconvolution_mock:
        with mock.patch("sys.argv", command_line_args):
            bayestme.cli.plot_deconvolution.main()

            plot_deconvolution_mock.assert_called_once_with(
                stdata=mock.ANY,
                output_dir=tmpdir,
                cell_type_names=["type 1", "type 2", "type 3", "type 4", "type 5"],
            )
