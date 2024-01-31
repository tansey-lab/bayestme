import os
import shutil
import tempfile

import numpy as np
import numpy.testing

import bayestme.common
import bayestme.data
from bayestme import data, synthetic_data, utils
from bayestme.synthetic_data import create_toy_deconvolve_result


def generate_toy_stdataset() -> data.SpatialExpressionDataset:
    raw_counts = np.array([[7, 1, 2], [8, 2, 3], [9, 3, 4]], dtype=np.int64)

    locations = np.array([(x, 0) for x in range(3)])

    tissue_mask = np.array([True for _ in range(3)])

    gene_names = np.array(["normal_ascii", "", "\t\t"])  # blank string,  # whitespace

    return data.SpatialExpressionDataset.from_arrays(
        raw_counts=raw_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=gene_names,
        layout=bayestme.common.Layout.SQUARE,
        edges=utils.get_edges(locations, bayestme.common.Layout.SQUARE),
        barcodes=np.array(["1", "2", "3"]),
    )


def test_serialize_deserialize_spatial_expression_dataset():
    dataset = generate_toy_stdataset()

    tmpdir = tempfile.mkdtemp()

    try:
        dataset.save(os.path.join(tmpdir, "data.h5"))

        new_dataset = data.SpatialExpressionDataset.read_h5(
            os.path.join(tmpdir, "data.h5")
        )

        np.testing.assert_array_equal(new_dataset.raw_counts, dataset.raw_counts)
        np.testing.assert_array_equal(new_dataset.positions, dataset.positions)
        np.testing.assert_array_equal(new_dataset.gene_names, dataset.gene_names)
        np.testing.assert_array_equal(new_dataset.tissue_mask, dataset.tissue_mask)
        assert new_dataset.layout == dataset.layout
    finally:
        shutil.rmtree(tmpdir)


def test_serialize_deserialize_deconvolution_results_dataset():
    n_samples = 100
    n_nodes = 25
    n_components = 4
    n_gene = 100
    cell_prob_trace = np.random.random((n_samples, n_nodes, n_components + 1))
    cell_num_trace = np.random.random((n_samples, n_nodes, n_components + 1))
    expression_trace = np.random.random((n_samples, n_components, n_gene))
    beta_trace = np.random.random((n_samples, n_components))
    reads_trace = np.random.random((n_samples, n_nodes, n_gene, n_components))
    losses = np.random.random((n_samples,))
    lam2 = 1000

    dataset = data.DeconvolutionResult(
        cell_prob_trace=cell_prob_trace,
        expression_trace=expression_trace,
        beta_trace=beta_trace,
        cell_num_trace=cell_num_trace,
        reads_trace=reads_trace,
        lam2=lam2,
        n_components=n_components,
        losses=losses,
    )

    tmpdir = tempfile.mkdtemp()

    try:
        dataset.save(os.path.join(tmpdir, "data.h5"))

        new_dataset = data.DeconvolutionResult.read_h5(os.path.join(tmpdir, "data.h5"))

        np.testing.assert_array_equal(
            new_dataset.cell_prob_trace, dataset.cell_prob_trace
        )
        np.testing.assert_array_equal(
            new_dataset.cell_num_trace, dataset.cell_num_trace
        )
        np.testing.assert_array_equal(
            new_dataset.expression_trace, dataset.expression_trace
        )
        np.testing.assert_array_equal(new_dataset.beta_trace, dataset.beta_trace)
        np.testing.assert_array_equal(new_dataset.losses, dataset.losses)
        assert new_dataset.lam2 == dataset.lam2
        assert new_dataset.n_components == dataset.n_components
    finally:
        shutil.rmtree(tmpdir)


def test_deconvolution_results_properties():
    rng = np.random.default_rng(1)
    n_samples = 100
    n_nodes = 25
    n_components = 4
    n_gene = 50
    cell_prob_trace = rng.random((n_samples, n_nodes, n_components))
    cell_num_trace = rng.random((n_samples, n_nodes, n_components))
    expression_trace = rng.random((n_samples, n_components, n_gene))
    beta_trace = rng.random((n_samples, n_components))
    reads_trace = rng.random((n_samples, n_nodes, n_gene, n_components))
    lam2 = 1000

    dataset = data.DeconvolutionResult(
        cell_prob_trace=cell_prob_trace,
        expression_trace=expression_trace,
        beta_trace=beta_trace,
        cell_num_trace=cell_num_trace,
        reads_trace=reads_trace,
        lam2=lam2,
        n_components=n_components,
    )

    assert dataset.omega.shape == (n_components, n_gene)
    assert (dataset.omega.sum(axis=0) == np.array(1.0)).sum() == 50

    assert dataset.omega_difference.shape == (n_components, n_gene)
    assert np.all(dataset.omega_difference <= 1.0)

    assert dataset.relative_expression.shape == (n_components, n_gene)
    assert np.all(dataset.relative_expression <= 1.0) and np.all(
        dataset.relative_expression >= -1.0
    )

    assert dataset.nb_probs.shape == (n_samples, n_nodes, n_gene)


def test_create_anndata_object():
    n_genes = 10
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=10, n_cols=10, n_genes=n_genes
    )

    gene_names = np.array([f"{i}" for i in range(n_genes)])

    adata = data.create_anndata_object(
        counts=bleed_counts,
        coordinates=locations,
        gene_names=gene_names,
        layout=bayestme.common.Layout.SQUARE,
        edges=utils.get_edges(
            locations[tissue_mask], layout=bayestme.common.Layout.SQUARE
        ),
        tissue_mask=tissue_mask,
    )

    np.testing.assert_array_equal(adata.X, bleed_counts)
    np.testing.assert_array_equal(adata.var_names, gene_names)
    np.testing.assert_array_equal(adata.obs[data.IN_TISSUE_ATTR], tissue_mask)
    np.testing.assert_array_equal(
        np.sort(np.array(adata.obsp[data.CONNECTIVITIES_ATTR].nonzero()).T, axis=0),
        np.sort(
            utils.get_edges(
                locations[tissue_mask], layout=bayestme.common.Layout.SQUARE
            ),
            axis=0,
        ),
    )
    assert adata.uns[data.LAYOUT_ATTR] == bayestme.common.Layout.SQUARE.name


def test_properties_work_without_obs_names():
    n_genes = 10
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=10, n_cols=10, n_genes=n_genes
    )

    gene_names = np.array([f"{i}" for i in range(n_genes)])
    adata = data.create_anndata_object(
        counts=bleed_counts,
        coordinates=locations,
        gene_names=gene_names,
        layout=bayestme.common.Layout.SQUARE,
        edges=utils.get_edges(locations, layout=bayestme.common.Layout.SQUARE),
        tissue_mask=tissue_mask,
    )

    dataset = data.SpatialExpressionDataset(adata)

    np.testing.assert_array_equal(dataset.counts, bleed_counts[tissue_mask])
    np.testing.assert_array_equal(dataset.positions_tissue, locations[tissue_mask])
    np.testing.assert_array_equal(dataset.n_spot_in, tissue_mask.sum())
    np.testing.assert_array_equal(dataset.raw_counts, bleed_counts)
    np.testing.assert_array_equal(dataset.positions, locations)
    np.testing.assert_array_equal(dataset.tissue_mask, tissue_mask)


def test_properties_work_with_obs_names():
    n_genes = 10
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=10, n_cols=10, n_genes=n_genes
    )

    gene_names = np.array([f"{i}" for i in range(n_genes)])
    barcodes = np.array([f"barcode{i}" for i in range(100)])
    adata = data.create_anndata_object(
        counts=bleed_counts,
        coordinates=locations,
        gene_names=gene_names,
        layout=bayestme.common.Layout.SQUARE,
        edges=utils.get_edges(locations, layout=bayestme.common.Layout.SQUARE),
        tissue_mask=tissue_mask,
        barcodes=barcodes,
    )

    dataset = data.SpatialExpressionDataset(adata)

    np.testing.assert_array_equal(dataset.counts, bleed_counts[tissue_mask])
    np.testing.assert_array_equal(dataset.positions_tissue, locations[tissue_mask])
    np.testing.assert_array_equal(dataset.n_spot_in, tissue_mask.sum())
    np.testing.assert_array_equal(dataset.raw_counts, bleed_counts)
    np.testing.assert_array_equal(dataset.positions, locations)
    np.testing.assert_array_equal(dataset.tissue_mask, tissue_mask)


def test_add_deconvolution_results_to_dataset():
    n_components = 3
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
        edges=bayestme.utils.get_edges(
            locations[tissue_mask], bayestme.common.Layout.SQUARE
        ),
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

    assert dataset.cell_type_probabilities is not None
    assert dataset.cell_type_counts is not None

    numpy.testing.assert_equal(
        dataset.cell_type_probabilities,
        deconvolve_results.cell_prob_trace.mean(axis=0),
    )

    numpy.testing.assert_equal(
        dataset.cell_type_counts,
        deconvolve_results.cell_num_trace.mean(axis=0),
    )

    assert dataset.n_cell_types == n_components


def test_add_deconvolution_results_to_dataset_with_obs_names():
    n_components = 3
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
        edges=bayestme.utils.get_edges(
            locations[tissue_mask], bayestme.common.Layout.SQUARE
        ),
        barcodes=barcodes,
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

    assert dataset.cell_type_probabilities is not None
    assert dataset.cell_type_counts is not None

    numpy.testing.assert_equal(
        dataset.cell_type_probabilities,
        deconvolve_results.cell_prob_trace.mean(axis=0),
    )

    numpy.testing.assert_equal(
        dataset.cell_type_counts,
        deconvolve_results.cell_num_trace.mean(axis=0),
    )

    assert dataset.n_cell_types == n_components
