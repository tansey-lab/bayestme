import shutil
import numpy as np
import tempfile
import os

import bayestme.synthetic_data
from bayestme import spatial_expression, bleeding_correction, utils, deconvolution, data
from unittest import mock


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

    dataset = bayestme.synthetic_data.generate_fake_stdataset(
        n_rows=12,
        n_cols=12,
        n_genes=n_genes)

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


def test_get_n_cell_correlation():
    result_for_correlated_arrays = spatial_expression.get_n_cell_correlation(
        np.array([1, 2, 3, 4, 5]),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    )
    np.random.seed(0)
    result_for_random = spatial_expression.get_n_cell_correlation(
        np.random.random(5),
        np.random.random(5)
    )

    assert result_for_correlated_arrays > result_for_random


def test_morans_i():
    square_dim = 9
    size = int(square_dim ** 2)

    # 1 0 1 0
    # 0 1 0 1
    # 1 0 1 0
    # 0 1 0 1
    checkerboard_w_pattern = np.arange(size) % 2

    # 0 1 1 1
    # 1 0 0 1
    # 0 1 0 1
    # 0 1 0 0
    np.random.seed(0)
    random_w_pattern = np.random.random(size) > 0.5

    # 1 1 1 1
    # 1 1 1 1
    # 0 0 0 0
    # 0 0 0 0
    half_size = int((square_dim ** 2) / 2)
    half_size_2 = size - half_size
    clustered_w_pattern = np.concatenate([np.ones(half_size), np.zeros(half_size_2)])

    positions = np.zeros((2, size))

    positions[1, :] = np.concatenate([np.arange(square_dim)] * square_dim)
    positions[0, :] = np.array(np.concatenate([np.array([i] * square_dim) for i in range(square_dim)]))

    positions = positions.astype(int)

    edges = utils.get_edges(positions, layout=data.Layout.SQUARE.value)

    clustered_value = spatial_expression.moran_i(edges, clustered_w_pattern)
    dispersed_value = spatial_expression.moran_i(edges, checkerboard_w_pattern)
    random_value = spatial_expression.moran_i(edges, random_w_pattern)

    assert dispersed_value < random_value < clustered_value
    assert dispersed_value < 0
    assert clustered_value > 0


def test_select_significant_spatial_programs():
    n_genes = 3
    n_components = 1
    n_samples = 10
    n_spatial_patterns = 1

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

    mock_setups = [
        [0.99, np.array([0.99, 0.99, 0.99]), np.array([0, 1, 2]), 0.99],  # Drop programs
        [0.1, np.array([0.1, 0.1, 0.1]), np.array([0, 1, 2]), 0.99],  # Drop programs
        [0.1, np.array([0.99, 0.99, 0.99]), np.array([]), 0.99],  # Drop programs
        [0.1, np.array([0.99, 0.99, 0.99]), np.array([0, 1, 2]), 0.01],  # Drop programs
        [0.1, np.array([0.99, 0.99, 0.99]), np.array([0, 1, 2]), 0.99]  # Accept programs
    ]

    expected_results = [
        [],
        [],
        [],
        [],
        [(0, 1)]
    ]

    for mock_setup, expected_result in zip(mock_setups, expected_results):
        with mock.patch('bayestme.spatial_expression.get_n_cell_correlation') as mock_get_n_cell_correlation:
            with mock.patch('bayestme.spatial_expression.get_proportion_of_spots_in_k_with_pattern_h_per_gene') as \
                    mock_get_proportion_of_spots_in_k_with_pattern_h_per_gene:
                with mock.patch(
                        'bayestme.spatial_expression.filter_pseudogenes_from_selection') as \
                        mock_filter_pseudogenes_from_selection:
                    with mock.patch('bayestme.spatial_expression.moran_i') as mock_moran_i:
                        cell_correlation, proportions, pseudogene_filter, morans_i = mock_setup

                        mock_get_n_cell_correlation.return_value = cell_correlation
                        mock_get_proportion_of_spots_in_k_with_pattern_h_per_gene.return_value = proportions

                        mock_filter_pseudogenes_from_selection.return_value = pseudogene_filter

                        mock_moran_i.return_value = morans_i

                        significant_spatial_programs = [(k, h) for (k, h, ids) in
                                                        spatial_expression.select_significant_spatial_programs(
                                                            stdata=dataset,
                                                            decon_result=deconvolution_results,
                                                            sde_result=sde_results,
                                                            tissue_threshold=1,
                                                            cell_correlation_threshold=0.5,
                                                            moran_i_score_threshold=0.9,
                                                            gene_spatial_pattern_proportion_threshold=0.95,
                                                            filter_pseudogenes=True
                                                        )]

                        assert sorted(significant_spatial_programs) == expected_result


def test_plot_spatial_pattern_with_legend():
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
        gene_names=np.array(['looong name', 'big big name', 'eirbgoewqugberf:erferf', '304ofh308fh3wf:sdfsdfsdr', 'erferfserf:44', 'fsdrfsdrgdsrv98dvfj', 'f34fawefc']),
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
        spatial_expression.plot_spatial_pattern_with_legend(
            stdata=dataset,
            decon_result=deconvolution_results,
            sde_result=sde_results,
            k=1,
            h=1,
            gene_ids=np.array([0, 1, 2]),
            output_file=os.path.join(tempdir, 'test.pdf'))
    finally:
        shutil.rmtree(tempdir)
