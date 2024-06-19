import os
import shutil
import tempfile

import numpy as np
import pandas

import bayestme.utils
import bayestme.common
import bayestme.data
import bayestme.marker_genes
import bayestme.synthetic_data
from bayestme import data
from bayestme.synthetic_data import (
    create_deconvolve_dataset,
    create_toy_deconvolve_result,
)


def test_detect_marker_genes_tight():
    n_components = 3
    n_marker = 2
    n_genes = 100
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
        edges=bayestme.utils.get_edges(locations, bayestme.common.Layout.SQUARE),
    )

    deconvolve_results = create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=n_components,
        n_samples=100,
        n_gene=dataset.n_gene,
    )

    marker_genes = bayestme.marker_genes.select_marker_genes(
        deconvolution_result=deconvolve_results,
        n_marker=n_marker,
        alpha=0.6,
        method=bayestme.marker_genes.MarkerGeneMethod.TIGHT,
    )

    assert len(marker_genes) == n_components
    for marker_gene_set in marker_genes:
        assert len(marker_gene_set) == n_marker


def test_detect_marker_genes_fdr():
    n_components = 3
    n_marker = 2
    n_genes = 100
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
        edges=bayestme.utils.get_edges(locations, bayestme.common.Layout.SQUARE),
    )

    deconvolve_results = create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=n_components,
        n_samples=100,
        n_gene=dataset.n_gene,
    )

    marker_genes = bayestme.marker_genes.select_marker_genes(
        deconvolution_result=deconvolve_results,
        n_marker=n_marker,
        alpha=0.99,
        method=bayestme.marker_genes.MarkerGeneMethod.FALSE_DISCOVERY_RATE,
    )

    assert len(marker_genes) == n_components
    for marker_gene_set in marker_genes:
        assert len(marker_gene_set) == n_marker


def test_detect_marker_genes_best_available():
    n_components = 3
    n_marker = 2
    n_genes = 100
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
        edges=bayestme.utils.get_edges(locations, bayestme.common.Layout.SQUARE),
    )

    deconvolve_results = create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=n_components,
        n_samples=100,
        n_gene=dataset.n_gene,
    )

    marker_genes = bayestme.marker_genes.select_marker_genes(
        deconvolution_result=deconvolve_results,
        n_marker=n_marker,
        alpha=0,
        method=bayestme.marker_genes.MarkerGeneMethod.BEST_AVAILABLE,
    )

    assert len(marker_genes) == n_components
    for marker_gene_set in marker_genes:
        assert len(marker_gene_set) == n_marker


def test_plot_marker_genes():
    tempdir = tempfile.mkdtemp()
    dataset = create_deconvolve_dataset(n_components=5)

    try:
        bayestme.marker_genes.plot_marker_genes(
            stdata=dataset,
            output_file=os.path.join(tempdir, "plot.pdf"),
            cell_type_labels=["B-Cell", "T-Cell", "Lymphoid", "Muscle", "Neuron"],
        )
    finally:
        shutil.rmtree(tempdir)


def test_add_marker_gene_results_to_dataset():
    n_components = 3
    n_marker = 2
    n_genes = 100
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12, n_cols=12, n_genes=n_genes
    )
    barcodes = np.array([f"barcode{i}" for i in range(12 * 12)])
    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(["{}".format(x) for x in range(n_genes)]),
        layout=bayestme.common.Layout.SQUARE,
        barcodes=barcodes,
        edges=bayestme.utils.get_edges(locations, bayestme.common.Layout.SQUARE),
    )

    deconvolve_results = create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=n_components,
        n_samples=100,
        n_gene=dataset.n_gene,
    )

    bayestme.data.add_deconvolution_results_to_dataset(
        stdata=dataset, result=deconvolve_results
    )

    marker_genes = bayestme.marker_genes.select_marker_genes(
        deconvolution_result=deconvolve_results, n_marker=n_marker, alpha=0.99
    )

    bayestme.marker_genes.add_marker_gene_results_to_dataset(
        stdata=dataset, marker_genes=marker_genes
    )

    for expected, observed in zip(marker_genes, dataset.marker_gene_indices):
        np.testing.assert_equal(expected, observed)


def test_add_marker_gene_results_to_dataset_with_obs_names():
    n_components = 3
    n_marker = 2
    n_genes = 100
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
        edges=bayestme.utils.get_edges(locations, bayestme.common.Layout.SQUARE),
    )

    deconvolve_results = create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=n_components,
        n_samples=100,
        n_gene=dataset.n_gene,
    )

    bayestme.data.add_deconvolution_results_to_dataset(
        stdata=dataset, result=deconvolve_results
    )

    marker_genes = bayestme.marker_genes.select_marker_genes(
        deconvolution_result=deconvolve_results, n_marker=n_marker, alpha=0.99
    )

    bayestme.marker_genes.add_marker_gene_results_to_dataset(
        stdata=dataset, marker_genes=marker_genes
    )

    for expected, observed in zip(marker_genes, dataset.marker_gene_indices):
        np.testing.assert_equal(expected, observed)


def test_create_top_gene_lists():
    n_components = 3
    n_marker = 2
    n_genes = 100
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
        edges=bayestme.utils.get_edges(locations, bayestme.common.Layout.SQUARE),
    )

    deconvolve_results = create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=n_components,
        n_samples=100,
        n_gene=dataset.n_gene,
    )

    bayestme.data.add_deconvolution_results_to_dataset(
        stdata=dataset, result=deconvolve_results
    )

    marker_genes = bayestme.marker_genes.select_marker_genes(
        deconvolution_result=deconvolve_results, n_marker=n_marker, alpha=0.99
    )

    bayestme.marker_genes.add_marker_gene_results_to_dataset(
        stdata=dataset, marker_genes=marker_genes
    )

    tempdir = tempfile.mkdtemp()

    bayestme.marker_genes.create_top_gene_lists(
        stdata=dataset,
        deconvolution_result=deconvolve_results,
        alpha=0.99,
        n_marker_genes=2,
        output_path=os.path.join(tempdir, "file.csv"),
    )

    result = pandas.read_csv(os.path.join(tempdir, "file.csv"))

    assert result.iloc[0].gene_name == marker_genes[0][0]

    assert (
        result.iloc[len(result) - 1].gene_name
        == marker_genes[len(marker_genes) - 1][len(marker_genes[-1]) - 1]
    )


def test_create_marker_gene_ranking_csvs():
    n_components = 3
    n_marker = 2
    n_genes = 100
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
        edges=bayestme.utils.get_edges(locations, bayestme.common.Layout.SQUARE),
    )

    deconvolve_results = create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=n_components,
        n_samples=100,
        n_gene=dataset.n_gene,
    )

    bayestme.data.add_deconvolution_results_to_dataset(
        stdata=dataset, result=deconvolve_results
    )

    marker_genes = bayestme.marker_genes.select_marker_genes(
        deconvolution_result=deconvolve_results, n_marker=n_marker, alpha=0.99
    )

    bayestme.marker_genes.add_marker_gene_results_to_dataset(
        stdata=dataset, marker_genes=marker_genes
    )

    tempdir = tempfile.mkdtemp()

    bayestme.marker_genes.create_marker_gene_ranking_csvs(
        stdata=dataset,
        deconvolution_result=deconvolve_results,
        output_dir=tempdir,
    )

    df = pandas.read_csv(os.path.join(tempdir, "relative_expression.csv"))
    assert df.shape == (dataset.n_gene, deconvolve_results.n_components + 1)
    df = pandas.read_csv(os.path.join(tempdir, "omega.csv"))
    assert df.shape == (dataset.n_gene, deconvolve_results.n_components + 1)
