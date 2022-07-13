import shutil
import numpy as np
import tempfile

import bayestme.synthetic_data
from bayestme import spatial_expression, bleeding_correction, utils, deconvolution, data


def generate_fake_deconvolve_results(n_samples, n_tissue_spots, n_components, n_genes) -> data.DeconvolutionResult:
    cell_prob_trace = np.random.random((n_samples, n_tissue_spots, n_components + 1))
    cell_num_trace = np.random.poisson(lam=10, size=(n_samples, n_tissue_spots, n_components + 1)).astype(np.float64)
    expression_trace = np.random.random((n_samples, n_components, n_genes))
    beta_trace = np.random.random((n_samples, n_components)) * 100.0
    reads_trace = np.random.poisson(lam=10, size=(n_samples, n_tissue_spots, n_genes, n_components)).astype(np.float64)

    return data.DeconvolutionResult(
        cell_prob_trace=cell_prob_trace,
        expression_trace=expression_trace,
        beta_trace=beta_trace,
        cell_num_trace=cell_num_trace,
        reads_trace=reads_trace,
        lam2=1000,
        n_components=n_components
    )


def generate_fake_sde_results(n_samples, n_genes, n_components, n_spatial_patterns,
                              n_spot_in) -> data.SpatialDifferentialExpressionResult:
    c_samples = np.random.random((n_samples, n_genes, n_components))
    gamma_samples = np.random.random((n_samples, n_components, n_spatial_patterns + 1))
    h_samples = np.zeros((n_samples, n_genes, n_components))
    spatial_pattern_choices = np.array(range(n_spatial_patterns))

    for component_id in range(n_components):
        for gene_id in range(n_genes):
            dominant_pattern = np.random.choice(spatial_pattern_choices)
            h_samples[:, gene_id, component_id] = dominant_pattern

    theta_samples = np.random.random((n_samples, n_spot_in, n_genes, n_components))
    v_samples = np.random.random((n_samples, n_genes, n_components))
    w_samples = np.random.random((n_samples, n_components, n_spatial_patterns + 1, n_spot_in))

    return data.SpatialDifferentialExpressionResult(
        c_samples=c_samples,
        gamma_samples=gamma_samples,
        h_samples=h_samples,
        theta_samples=theta_samples,
        v_samples=v_samples,
        w_samples=w_samples)


def test_spatial_detection():
    n_genes = 7
    n_components = 3
    n_samples = 10
    n_spatial_patterns = 10

    locations, tissue_mask, true_rates, true_counts, bleed_counts = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12,
        n_cols=12,
        n_genes=n_genes)

    dataset = data.SpatialExpressionDataset(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations.T,
        gene_names=np.array(['{}'.format(x) for x in range(n_genes)]),
        layout=data.Layout.SQUARE
    )

    deconvolution_results = generate_fake_deconvolve_results(
        n_samples=n_samples,
        n_tissue_spots=dataset.n_spot_in,
        n_components=n_components,
        n_genes=n_genes)

    sde_result = spatial_expression.run_spatial_expression(
        dataset=dataset,
        deconvolve_results=deconvolution_results,
        n_spatial_patterns=10,
        n_samples=n_samples,
        n_burn=1,
        n_thin=1,
        n_cell_min=5,
        alpha0=1,
        prior_var=100.0,
        lam2=1000,
        simple=True)

    assert sde_result.c_samples.shape == (n_samples, n_genes, n_components)
    assert sde_result.gamma_samples.shape == (n_samples, n_components, n_spatial_patterns + 1)
    assert sde_result.h_samples.shape == (n_samples, n_genes, n_components)
    assert sde_result.theta_samples.shape == (n_samples, dataset.n_spot_in, n_genes, n_components)
    assert sde_result.v_samples.shape == (n_samples, n_genes, n_components)
    assert sde_result.w_samples.shape == (n_samples, n_components, n_spatial_patterns + 1, dataset.n_spot_in)


def test_select_significant_spatial_programs():
    n_genes = 7
    n_components = 3
    n_samples = 10
    n_spatial_patterns = 10

    dataset = bayestme.synthetic_data.generate_fake_stdataset(
        n_rows=50,
        n_cols=50,
        n_genes=n_genes)

    deconvolution_results = generate_fake_deconvolve_results(
        n_samples=n_samples,
        n_tissue_spots=dataset.n_spot_in,
        n_components=n_components,
        n_genes=n_genes)

    sde_results = generate_fake_sde_results(
        n_samples=n_samples,
        n_genes=n_genes,
        n_components=n_components,
        n_spatial_patterns=n_spatial_patterns,
        n_spot_in=dataset.n_spot_in)

    result = [_ for _ in spatial_expression.select_significant_spatial_programs(
        stdata=dataset,
        decon_result=deconvolution_results,
        sde_result=sde_results,
    )]


def test_plot_spatial_patterns():
    n_genes = 7
    n_components = 3
    n_samples = 10
    n_spatial_patterns = 10

    locations, tissue_mask, true_rates, true_counts, bleed_counts = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=50,
        n_cols=50,
        n_genes=n_genes)

    dataset = data.SpatialExpressionDataset(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations.T,
        gene_names=np.array(['{}'.format(x) for x in range(n_genes)]),
        layout=data.Layout.SQUARE
    )
    deconvolution_results = generate_fake_deconvolve_results(
        n_samples=n_samples,
        n_tissue_spots=dataset.n_spot_in,
        n_components=n_components,
        n_genes=n_genes)

    sde_results = generate_fake_sde_results(
        n_samples=n_samples,
        n_genes=n_genes,
        n_components=n_components,
        n_spatial_patterns=n_spatial_patterns,
        n_spot_in=dataset.n_spot_in)

    tempdir = tempfile.mkdtemp()

    try:
        spatial_expression.plot_spatial_patterns(
            stdata=dataset,
            decon_result=deconvolution_results,
            sde_result=sde_results,
            output_dir=tempdir,
            cell_correlation_threshold=100,
            moran_i_score_threshold=-1,
            gene_spatial_pattern_proportion_threshold=-1)
    finally:
        shutil.rmtree(tempdir)
