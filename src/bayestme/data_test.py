import numpy as np
import tempfile
import os
import shutil

from bayestme import data, synthetic_data, utils


def generate_toy_stdataset() -> data.SpatialExpressionDataset:
    raw_counts = np.array(
        [[7, 1, 2],
         [8, 2, 3],
         [9, 3, 4]], dtype=np.int64
    )

    locations = np.array([
        (x, 0) for x in range(3)
    ])

    tissue_mask = np.array([True for _ in range(3)])

    gene_names = np.array(['normal_ascii',
                           '',  # blank string,
                           '\t\t'  # whitespace
                           ])

    return data.SpatialExpressionDataset.from_arrays(
        raw_counts=raw_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=gene_names,
        layout=data.Layout.SQUARE
    )


def test_serialize_deserialize_spatial_expression_dataset():
    dataset = generate_toy_stdataset()

    tmpdir = tempfile.mkdtemp()

    try:
        dataset.save(os.path.join(tmpdir, 'data.h5'))

        new_dataset = data.SpatialExpressionDataset.read_h5(os.path.join(tmpdir, 'data.h5'))

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
    lam2 = 1000

    dataset = data.DeconvolutionResult(
        cell_prob_trace=cell_prob_trace,
        expression_trace=expression_trace,
        beta_trace=beta_trace,
        cell_num_trace=cell_num_trace,
        reads_trace=reads_trace,
        lam2=lam2,
        n_components=n_components
    )

    tmpdir = tempfile.mkdtemp()

    try:
        dataset.save(os.path.join(tmpdir, 'data.h5'))

        new_dataset = data.DeconvolutionResult.read_h5(os.path.join(tmpdir, 'data.h5'))

        np.testing.assert_array_equal(new_dataset.cell_prob_trace, dataset.cell_prob_trace)
        np.testing.assert_array_equal(new_dataset.cell_num_trace, dataset.cell_num_trace)
        np.testing.assert_array_equal(new_dataset.expression_trace, dataset.expression_trace)
        np.testing.assert_array_equal(new_dataset.beta_trace, dataset.beta_trace)

        assert new_dataset.lam2 == dataset.lam2
        assert new_dataset.n_components == dataset.n_components
    finally:
        shutil.rmtree(tmpdir)


def test_create_anndata_object():
    n_genes = 10
    locations, tissue_mask, true_rates, true_counts, bleed_counts = synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=10, n_cols=10, n_genes=n_genes)

    gene_names = np.array([f'{i}' for i in range(n_genes)])

    adata = data.create_anndata_object(counts=bleed_counts,
                                       coordinates=locations,
                                       gene_names=gene_names,
                                       layout=data.Layout.SQUARE,
                                       tissue_mask=tissue_mask)

    np.testing.assert_array_equal(adata.X, bleed_counts)
    np.testing.assert_array_equal(adata.var_names, gene_names)
    np.testing.assert_array_equal(adata.obs[data.IN_TISSUE_ATTR], tissue_mask)
    np.testing.assert_array_equal(
        np.sort(np.array(adata.obsp[data.CONNECTIVITIES_ATTR].nonzero()).T, axis=0),
        np.sort(utils.get_edges(locations[tissue_mask], layout=data.Layout.SQUARE.value), axis=0)
    )
    assert adata.uns[data.LAYOUT_ATTR] == data.Layout.SQUARE.name

