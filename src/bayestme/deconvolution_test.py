import shutil
import tempfile
import numpy as np
import os
import pandas

import bayestme.synthetic_data
from bayestme import bleeding_correction, utils, deconvolution, data


def create_toy_deconvolve_result(
        n_nodes: int,
        n_components: int,
        n_samples: int,
        n_gene: int) -> data.DeconvolutionResult:
    return data.DeconvolutionResult(
        lam2=1000,
        n_components=n_components,
        cell_num_trace=np.random.random((n_samples, n_nodes, n_components + 1)),
        cell_prob_trace=np.random.random((n_samples, n_nodes, n_components + 1)),
        expression_trace=np.random.random((n_samples, n_components, n_gene)),
        beta_trace=np.random.random((n_samples, n_components)),
        reads_trace=np.random.random((n_samples, n_nodes, n_gene, n_components))
    )


def test_deconvolve():
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=15,
        n_cols=15,
        n_genes=5)

    edges = utils.get_edges(locations[tissue_mask, :], 2)

    n_samples = 3
    lam2 = 1000
    n_components = 3
    n_nodes = tissue_mask.sum()
    n_gene = 3

    result: data.DeconvolutionResult = deconvolution.deconvolve(
        true_counts[tissue_mask],
        edges,
        n_gene=3,
        n_components=3,
        lam2=1000,
        n_samples=3,
        n_burnin=1,
        n_thin=1,
        random_seed=0,
        bkg=False,
        lda=False
    )

    assert result.lam2 == lam2
    assert result.n_components == n_components
    assert result.cell_prob_trace.shape == (n_samples, n_nodes, n_components + 1)
    assert result.cell_num_trace.shape == (n_samples, n_nodes, n_components + 1)
    assert result.expression_trace.shape == (n_samples, n_components, n_gene)
    assert result.beta_trace.shape == (n_samples, n_components)
    assert result.reads_trace.shape == (n_samples, n_nodes, n_gene, n_components)


def test_detect_marker_genes():
    n_components = 3
    n_marker = 2
    n_genes = 100
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12,
        n_cols=12,
        n_genes=n_genes)

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(['{}'.format(x) for x in range(n_genes)]),
        layout=data.Layout.SQUARE)

    deconvolve_results = create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=n_components,
        n_samples=100,
        n_gene=dataset.n_gene)

    marker_genes, omega_difference = deconvolution.detect_marker_genes(
        deconvolution_result=deconvolve_results, n_marker=n_marker, alpha=0.99)

    assert len(marker_genes) == n_components
    for marker_gene_set in marker_genes:
        assert len(marker_gene_set) == n_marker
    assert omega_difference.shape == (n_components, dataset.n_gene)


def test_plot_marker_genes():
    tempdir = tempfile.mkdtemp()
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

    deconvolve_results = create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=5,
        n_samples=100,
        n_gene=dataset.n_gene)

    difference = np.random.random((n_marker_genes, n_genes))

    try:
        deconvolution.plot_marker_genes(
            marker_genes=[np.array([1]), np.array([2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9, 10]),
                          np.array([11, 12, 13, 14, 15])],
            difference=difference,
            stdata=dataset,
            deconvolution_results=deconvolve_results,
            output_file=os.path.join(tempdir, 'plot.pdf'),
            cell_type_labels=['B-Cell', 'T-Cell', 'Lymphoid', 'Muscle', 'Neuron']
        )
    finally:
        shutil.rmtree(tempdir)


def test_deconvolve_plots():
    tempdir = tempfile.mkdtemp()
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

    deconvolve_results = create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=5,
        n_samples=100,
        n_gene=dataset.n_gene)

    try:
        deconvolution.plot_deconvolution(
            stdata=dataset,
            deconvolution_result=deconvolve_results,
            output_dir=tempdir,
            n_marker_genes=n_marker_genes,
            alpha=0.99
        )
    finally:
        shutil.rmtree(tempdir)


def test_deconvolve_plots_with_cell_type_names():
    tempdir = tempfile.mkdtemp()
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

    deconvolve_results = create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=5,
        n_samples=100,
        n_gene=dataset.n_gene)

    try:
        deconvolution.plot_deconvolution(
            stdata=dataset,
            deconvolution_result=deconvolve_results,
            output_dir=tempdir,
            n_marker_genes=n_marker_genes,
            alpha=0.99,
            cell_type_names=['type1', 'banana', 'threeve', 'quattro', 'ISPC']
        )
    finally:
        shutil.rmtree(tempdir)


def test_create_top_gene_lists():
    n_components = 3
    n_marker = 2
    n_genes = 100
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12,
        n_cols=12,
        n_genes=n_genes)

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(['{}'.format(x) for x in range(n_genes)]),
        layout=data.Layout.SQUARE)

    deconvolve_results = create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=n_components,
        n_samples=100,
        n_gene=dataset.n_gene)

    marker_genes, omega_difference = deconvolution.detect_marker_genes(
        deconvolution_result=deconvolve_results, n_marker=n_marker, alpha=0.99)

    tempdir = tempfile.mkdtemp()

    deconvolution.create_top_gene_lists(
        stdata=dataset,
        deconvolution_result=deconvolve_results,
        alpha=0.99,
        n_marker_genes=2,
        output_path=os.path.join(tempdir, 'file.csv'))

    result = pandas.read_csv(os.path.join(tempdir, 'file.csv'))

    assert result.iloc[0].gene_name == marker_genes[0, 0]

    assert result.iloc[len(result) - 1].gene_name == marker_genes[marker_genes.shape[0] - 1, marker_genes.shape[1] - 1]


def test_load_phi_truth():
    example_data = {'0': {'LINC01409': 6.20854181530519e-06,
                          'LINC01128': 7.94814125442517e-06,
                          'LINC00115': 4.54911539806641e-06,
                          'FAM41C': 0.0,
                          'SAMD11': 0.0},
                    '1': {'LINC01409': 1.49995044600798e-05,
                          'LINC01128': 2.35076069675239e-05,
                          'LINC00115': 2.27916942034948e-06,
                          'FAM41C': 0.0,
                          'SAMD11': 1.40427550443168e-06},
                    '2': {'LINC01409': 2.97631602698784e-06,
                          'LINC01128': 1.45160772961017e-05,
                          'LINC00115': 1.46704940943111e-06,
                          'FAM41C': 7.18370422770737e-07,
                          'SAMD11': 0.0},
                    '3': {'LINC01409': 7.58645413781156e-06,
                          'LINC01128': 8.1726936453575e-06,
                          'LINC00115': 6.12543772975621e-07,
                          'FAM41C': 0.0,
                          'SAMD11': 0.0},
                    '4': {'LINC01409': 1.87487192566961e-06,
                          'LINC01128': 1.70604741811557e-05,
                          'LINC00115': 3.46678185392759e-06,
                          'FAM41C': 0.0,
                          'SAMD11': 7.77426624554906e-07},
                    '5': {'LINC01409': 7.83310236784797e-06,
                          'LINC01128': 9.96282875190567e-06,
                          'LINC00115': 4.87937499885119e-06,
                          'FAM41C': 9.4615520015824e-07,
                          'SAMD11': 0.0},
                    '6': {'LINC01409': 6.68166666509352e-06,
                          'LINC01128': 2.59127189953248e-05,
                          'LINC00115': 0.0,
                          'FAM41C': 0.0,
                          'SAMD11': 0.0},
                    '7': {'LINC01409': 3.87608783680169e-06,
                          'LINC01128': 1.35800435754236e-05,
                          'LINC00115': 9.97461460582817e-06,
                          'FAM41C': 0.0,
                          'SAMD11': 0.0},
                    '8': {'LINC01409': 1.05828228135006e-05,
                          'LINC01128': 4.36250687094832e-06,
                          'LINC00115': 4.72581079234104e-06,
                          'FAM41C': 0.0,
                          'SAMD11': 0.0}}

    tmpdir = tempfile.mkdtemp()

    locations, tissue_mask, true_rates, true_counts, bleed_counts = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12,
        n_cols=12,
        n_genes=3)

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(['LINC01409', 'LINC01128', 'LINC00115']),
        layout=data.Layout.SQUARE)

    pandas.DataFrame(example_data).to_csv(os.path.join(tmpdir, "test.csv"), index=True)

    result = deconvolution.load_expression_truth(dataset, os.path.join(tmpdir, "test.csv"))

    assert result.shape == (9, 3)

    np.testing.assert_almost_equal(result.sum(axis=1), np.ones(9))
