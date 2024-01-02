import os
import tempfile
from unittest import mock

import numpy as np

import bayestme.cli.plot_spatial_expression
import bayestme.common
import bayestme.synthetic_data
from bayestme import data, spatial_expression_test


def test_plot_spatial_expression():
    n_genes = 7
    n_components = 3
    n_samples = 10
    n_spatial_patterns = 10

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
        gene_names=np.array(["{}".format(x) for x in range(n_genes)]),
        layout=bayestme.common.Layout.SQUARE,
    )
    deconvolution_results = spatial_expression_test.generate_fake_deconvolve_results(
        n_samples=n_samples,
        n_tissue_spots=dataset.n_spot_in,
        n_components=n_components,
        n_genes=n_genes,
    )

    sde_results = spatial_expression_test.generate_fake_sde_results(
        n_samples=n_samples,
        n_genes=n_genes,
        n_components=n_components,
        n_spatial_patterns=n_spatial_patterns,
        n_spot_in=dataset.n_spot_in,
    )

    tmpdir = tempfile.mkdtemp()

    stdata_fn = os.path.join(tmpdir, "data.h5")
    deconvolve_results_fn = os.path.join(tmpdir, "deconvolve.h5")
    sde_results_fn = os.path.join(tmpdir, "sde.h5")
    dataset.save(stdata_fn)
    deconvolution_results.save(deconvolve_results_fn)
    sde_results.save(sde_results_fn)

    command_line_args = [
        "plot_spatial_expression",
        "--adata",
        stdata_fn,
        "--deconvolution-result",
        deconvolve_results_fn,
        "--sde-result",
        sde_results_fn,
        "--moran-i-score-threshold",
        "0.9",
        "--tissue-threshold",
        "5",
        "--gene-spatial-pattern-proportion-threshold",
        "0.95",
        "--output-dir",
        tmpdir,
    ]

    with mock.patch(
        "bayestme.spatial_expression.plot_significant_spatial_patterns"
    ) as plot_spatial_patterns_mock:
        with mock.patch("sys.argv", command_line_args):
            bayestme.cli.plot_spatial_expression.main()

            plot_spatial_patterns_mock.assert_called_once_with(
                stdata=mock.ANY,
                decon_result=mock.ANY,
                sde_result=mock.ANY,
                output_dir=tmpdir,
                cell_type_names=None,
                moran_i_score_threshold=0.9,
                tissue_threshold=5,
                gene_spatial_pattern_proportion_threshold=0.95,
            )


def test_plot_spatial_expression_with_cell_type_names():
    n_genes = 7
    n_components = 3
    n_samples = 10
    n_spatial_patterns = 10

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
        gene_names=np.array(["{}".format(x) for x in range(n_genes)]),
        layout=bayestme.common.Layout.SQUARE,
    )
    deconvolution_results = spatial_expression_test.generate_fake_deconvolve_results(
        n_samples=n_samples,
        n_tissue_spots=dataset.n_spot_in,
        n_components=n_components,
        n_genes=n_genes,
    )

    sde_results = spatial_expression_test.generate_fake_sde_results(
        n_samples=n_samples,
        n_genes=n_genes,
        n_components=n_components,
        n_spatial_patterns=n_spatial_patterns,
        n_spot_in=dataset.n_spot_in,
    )

    tmpdir = tempfile.mkdtemp()

    stdata_fn = os.path.join(tmpdir, "data.h5")
    deconvolve_results_fn = os.path.join(tmpdir, "deconvolve.h5")
    sde_results_fn = os.path.join(tmpdir, "sde.h5")
    dataset.save(stdata_fn)
    deconvolution_results.save(deconvolve_results_fn)
    sde_results.save(sde_results_fn)

    command_line_args = [
        "plot_spatial_expression",
        "--adata",
        stdata_fn,
        "--deconvolution-result",
        deconvolve_results_fn,
        "--sde-result",
        sde_results_fn,
        "--output-dir",
        tmpdir,
        "--moran-i-score-threshold",
        "0.9",
        "--tissue-threshold",
        "5",
        "--gene-spatial-pattern-proportion-threshold",
        "0.95",
        "--cell-type-names",
        "type 1, type 2, type 3",
    ]

    with mock.patch(
        "bayestme.spatial_expression.plot_significant_spatial_patterns"
    ) as plot_spatial_patterns_mock:
        with mock.patch("sys.argv", command_line_args):
            bayestme.cli.plot_spatial_expression.main()

            plot_spatial_patterns_mock.assert_called_once_with(
                stdata=mock.ANY,
                decon_result=mock.ANY,
                sde_result=mock.ANY,
                output_dir=tmpdir,
                cell_type_names=["type 1", "type 2", "type 3"],
                moran_i_score_threshold=0.9,
                tissue_threshold=5,
                gene_spatial_pattern_proportion_threshold=0.95,
            )
