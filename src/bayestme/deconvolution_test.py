import shutil
import tempfile
import numpy as np

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

    assert marker_genes.shape == (n_components, n_marker)
    assert omega_difference.shape == (n_components, dataset.n_gene)


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
