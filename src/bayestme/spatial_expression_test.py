import copy
import os
import shutil
import tempfile
from unittest import mock

import numpy as np
import numpy.testing
from numpy import testing

import bayestme.synthetic_data
from bayestme import spatial_expression, utils, data


def generate_fake_deconvolve_results(
    n_samples, n_tissue_spots, n_components, n_genes
) -> data.DeconvolutionResult:
    cell_prob_trace = np.random.random((n_samples, n_tissue_spots, n_components + 1))
    cell_num_trace = np.random.poisson(
        lam=10, size=(n_samples, n_tissue_spots, n_components + 1)
    ).astype(np.float64)
    expression_trace = np.random.random((n_samples, n_components, n_genes))
    beta_trace = np.random.random((n_samples, n_components)) * 100.0
    reads_trace = np.random.poisson(
        lam=10, size=(n_samples, n_tissue_spots, n_genes, n_components)
    ).astype(np.float64)

    return data.DeconvolutionResult(
        cell_prob_trace=cell_prob_trace,
        expression_trace=expression_trace,
        beta_trace=beta_trace,
        cell_num_trace=cell_num_trace,
        reads_trace=reads_trace,
        lam2=1000,
        n_components=n_components,
    )


def generate_fake_sde_results(
    n_samples, n_genes, n_components, n_spatial_patterns, n_spot_in
) -> data.SpatialDifferentialExpressionResult:
    c_samples = np.random.random((n_samples, n_genes, n_components))
    gamma_samples = np.random.random((n_samples, n_components, n_spatial_patterns + 1))
    h_samples = np.zeros((n_samples, n_genes, n_components))
    spatial_pattern_choices = np.array(range(n_spatial_patterns))

    for component_id in range(n_components):
        for gene_id in range(n_genes):
            dominant_pattern = np.random.choice(spatial_pattern_choices)
            h_samples[:, gene_id, component_id] = dominant_pattern

    theta_samples = np.random.random((n_samples, n_spot_in, n_genes, n_components))
    v_samples = (np.random.random((n_samples, n_genes, n_components)) - 0.5) * 2
    w_samples = np.random.random(
        (n_samples, n_components, n_spatial_patterns + 1, n_spot_in)
    )

    return data.SpatialDifferentialExpressionResult(
        c_samples=c_samples,
        gamma_samples=gamma_samples,
        h_samples=h_samples,
        theta_samples=theta_samples,
        v_samples=v_samples,
        w_samples=w_samples,
    )


def test_spatial_detection():
    n_genes = 7
    n_components = 3
    n_samples = 10
    n_spatial_patterns = 10

    dataset = bayestme.synthetic_data.generate_fake_stdataset(
        n_rows=12, n_cols=12, n_genes=n_genes
    )

    deconvolution_results = generate_fake_deconvolve_results(
        n_samples=n_samples,
        n_tissue_spots=dataset.n_spot_in,
        n_components=n_components,
        n_genes=n_genes,
    )

    alpha = np.ones(n_spatial_patterns + 1)
    alpha[0] = 10
    alpha[1:] = 1 / n_spatial_patterns

    n_nodes = dataset.n_spot_in
    n_signals = dataset.n_gene
    prior_vars = np.repeat(100.0, 2)

    sde = spatial_expression.SpatialDifferentialExpression(
        n_cell_types=deconvolution_results.n_components,
        n_spatial_patterns=n_spatial_patterns,
        n_nodes=n_nodes,
        n_signals=n_signals,
        edges=dataset.edges,
        alpha=alpha,
        prior_vars=prior_vars,
        lam2=1000.0,
    )

    sde.initialize()

    sde_result = spatial_expression.run_spatial_expression(
        sde=sde,
        deconvolve_results=deconvolution_results,
        n_samples=n_samples,
        n_burn=1,
        n_thin=1,
        n_cell_min=5,
        simple=True,
    )

    assert sde_result.c_samples.shape == (n_samples, n_genes, n_components)
    assert sde_result.gamma_samples.shape == (
        n_samples,
        n_components,
        n_spatial_patterns + 1,
    )
    assert sde_result.h_samples.shape == (n_samples, n_genes, n_components)
    assert sde_result.theta_samples.shape == (
        n_samples,
        dataset.n_spot_in,
        n_genes,
        n_components,
    )
    assert sde_result.v_samples.shape == (n_samples, n_genes, n_components)
    assert sde_result.w_samples.shape == (
        n_samples,
        n_components,
        n_spatial_patterns + 1,
        dataset.n_spot_in,
    )


def test_spatial_detection_seed_determinism():
    n_genes = 7
    n_components = 3
    n_samples = 10
    n_spatial_patterns = 10

    dataset = bayestme.synthetic_data.generate_fake_stdataset(
        n_rows=12, n_cols=12, n_genes=n_genes
    )

    deconvolution_results = generate_fake_deconvolve_results(
        n_samples=n_samples,
        n_tissue_spots=dataset.n_spot_in,
        n_components=n_components,
        n_genes=n_genes,
    )

    alpha = np.ones(n_spatial_patterns + 1)
    alpha[0] = 10
    alpha[1:] = 1 / n_spatial_patterns

    n_nodes = dataset.n_spot_in
    n_signals = dataset.n_gene
    prior_vars = np.repeat(100.0, 2)

    sde_1 = spatial_expression.SpatialDifferentialExpression(
        n_cell_types=deconvolution_results.n_components,
        n_spatial_patterns=n_spatial_patterns,
        n_nodes=n_nodes,
        n_signals=n_signals,
        edges=dataset.edges,
        alpha=alpha,
        prior_vars=prior_vars,
        lam2=1000.0,
        rng=np.random.default_rng(seed=99),
    )
    sde_1.initialize()

    sde_1_state = sde_1.get_state()

    tmpdir = tempfile.mkdtemp()

    sde_1_state.save(os.path.join(tmpdir, "state.h5"))

    sde_1_state = data.SpatialDifferentialExpressionSamplerState.read_h5(
        os.path.join(tmpdir, "state.h5")
    )

    sde_2 = spatial_expression.SpatialDifferentialExpression.load_from_state(
        sde_1_state
    )

    assert sde_2.rng.bit_generator.state == sde_1.rng.bit_generator.state

    sde_result_1 = spatial_expression.run_spatial_expression(
        sde=sde_1,
        deconvolve_results=deconvolution_results,
        n_samples=n_samples,
        n_burn=1,
        n_thin=1,
        n_cell_min=5,
        simple=True,
    )

    sde_result_2 = spatial_expression.run_spatial_expression(
        sde=sde_2,
        deconvolve_results=deconvolution_results,
        n_samples=n_samples,
        n_burn=1,
        n_thin=1,
        n_cell_min=5,
        simple=True,
    )

    assert sde_2.rng.bit_generator.state == sde_1.rng.bit_generator.state
    np.testing.assert_array_equal(sde_result_1.c_samples, sde_result_2.c_samples)
    np.testing.assert_array_equal(sde_result_1.w_samples, sde_result_2.w_samples)
    np.testing.assert_array_equal(sde_result_1.v_samples, sde_result_2.v_samples)
    np.testing.assert_array_equal(
        sde_result_1.theta_samples, sde_result_2.theta_samples
    )
    np.testing.assert_array_equal(sde_result_1.h_samples, sde_result_2.h_samples)
    np.testing.assert_array_equal(
        sde_result_1.gamma_samples, sde_result_2.gamma_samples
    )


def test_spatial_detection_seed_checkpointing():
    tmpdir = tempfile.mkdtemp()
    n_genes = 7
    n_components = 3
    n_samples = 10
    n_spatial_patterns = 10

    dataset = bayestme.synthetic_data.generate_fake_stdataset(
        n_rows=12, n_cols=12, n_genes=n_genes
    )

    deconvolution_results = generate_fake_deconvolve_results(
        n_samples=n_samples,
        n_tissue_spots=dataset.n_spot_in,
        n_components=n_components,
        n_genes=n_genes,
    )

    alpha = np.ones(n_spatial_patterns + 1)
    alpha[0] = 10
    alpha[1:] = 1 / n_spatial_patterns

    n_nodes = dataset.n_spot_in
    n_signals = dataset.n_gene
    prior_vars = np.repeat(100.0, 2)

    sde_1 = spatial_expression.SpatialDifferentialExpression(
        n_cell_types=deconvolution_results.n_components,
        n_spatial_patterns=n_spatial_patterns,
        n_nodes=n_nodes,
        n_signals=n_signals,
        edges=dataset.edges,
        alpha=alpha,
        prior_vars=prior_vars,
        lam2=1000.0,
        rng=np.random.default_rng(seed=99),
    )

    sde_1.initialize()

    ncell_min = 5
    cell_type_filter = (
        deconvolution_results.cell_num_trace[:, :, 1:].mean(axis=0) > ncell_min
    ).T
    rate = np.array(
        [
            deconvolution_results.beta_trace[i][:, None]
            * deconvolution_results.expression_trace[i]
            for i in range(deconvolution_results.cell_num_trace.shape[0])
        ]
    )
    reads = deconvolution_results.reads_trace.mean(axis=0).astype(int)
    lambdas = (
        deconvolution_results.cell_num_trace.mean(axis=0)[:, 1:, None]
        * rate.mean(axis=0)[None]
    )
    Y_igk = reads
    n_obs_vector = np.transpose(lambdas, [0, 2, 1])

    sde_1.sample(n_obs_vector, Y_igk, cell_type_filter)

    sde_1_state = sde_1.get_state()

    sde_1_state.save(os.path.join(tmpdir, "state.h5"))

    sde_1_state = data.SpatialDifferentialExpressionSamplerState.read_h5(
        os.path.join(tmpdir, "state.h5")
    )

    sde_2 = spatial_expression.SpatialDifferentialExpression.load_from_state(
        sde_1_state
    )

    sample_spatial_weights_mock = mock.MagicMock()
    sample_spatial_weights_mock.side_effect = ValueError("test")

    with mock.patch.object(
        sde_1, "sample_spatial_weights", sample_spatial_weights_mock
    ):
        try:
            sde_1.sample(n_obs_vector, Y_igk, cell_type_filter)
        except ValueError:
            pass

    sde_1.reset_to_checkpoint()

    W, C, Gamma, H, V, Theta = sde_1.sample(n_obs_vector, Y_igk, cell_type_filter)

    W_1, C_1, Gamma_1, H_1, V_1, Theta_1 = sde_2.sample(
        n_obs_vector, Y_igk, cell_type_filter
    )

    numpy.testing.assert_equal(W, W_1)
    numpy.testing.assert_equal(C, C_1)
    numpy.testing.assert_equal(Gamma, Gamma_1)
    numpy.testing.assert_equal(H, H_1)
    numpy.testing.assert_equal(V, V_1)
    numpy.testing.assert_equal(Theta, Theta_1)

    shutil.rmtree(tmpdir)


def test_spatial_detection_sampler_state_serialization_equivalency():
    tmpdir = tempfile.mkdtemp()
    n_genes = 7
    n_components = 3
    n_samples = 10
    n_spatial_patterns = 10

    dataset = bayestme.synthetic_data.generate_fake_stdataset(
        n_rows=12, n_cols=12, n_genes=n_genes
    )

    deconvolution_results = generate_fake_deconvolve_results(
        n_samples=n_samples,
        n_tissue_spots=dataset.n_spot_in,
        n_components=n_components,
        n_genes=n_genes,
    )

    alpha = np.ones(n_spatial_patterns + 1)
    alpha[0] = 10
    alpha[1:] = 1 / n_spatial_patterns

    n_nodes = dataset.n_spot_in
    n_signals = dataset.n_gene
    prior_vars = np.repeat(100.0, 2)

    sde_1 = spatial_expression.SpatialDifferentialExpression(
        n_cell_types=deconvolution_results.n_components,
        n_spatial_patterns=n_spatial_patterns,
        n_nodes=n_nodes,
        n_signals=n_signals,
        edges=dataset.edges,
        alpha=alpha,
        prior_vars=prior_vars,
        lam2=1000.0,
        rng=np.random.default_rng(seed=99),
    )

    sde_2 = spatial_expression.SpatialDifferentialExpression(
        n_cell_types=deconvolution_results.n_components,
        n_spatial_patterns=n_spatial_patterns,
        n_nodes=n_nodes,
        n_signals=n_signals,
        edges=dataset.edges,
        alpha=alpha,
        prior_vars=prior_vars,
        lam2=1000.0,
        rng=np.random.default_rng(seed=99),
    )

    sde_1.initialize()
    sde_2.initialize()

    ncell_min = 5
    cell_type_filter = (
        deconvolution_results.cell_num_trace[:, :, 1:].mean(axis=0) > ncell_min
    ).T
    rate = np.array(
        [
            deconvolution_results.beta_trace[i][:, None]
            * deconvolution_results.expression_trace[i]
            for i in range(deconvolution_results.cell_num_trace.shape[0])
        ]
    )
    reads = deconvolution_results.reads_trace.mean(axis=0).astype(int)
    lambdas = (
        deconvolution_results.cell_num_trace.mean(axis=0)[:, 1:, None]
        * rate.mean(axis=0)[None]
    )
    Y_igk = reads
    n_obs_vector = np.transpose(lambdas, [0, 2, 1])

    sde_1_outputs = []
    sde_2_outputs = []

    for i in range(5):
        sde_1_outputs.append(
            sde_1.sample(
                copy.deepcopy(n_obs_vector),
                copy.deepcopy(Y_igk),
                copy.deepcopy(cell_type_filter),
            )
        )

        sde_2_state = sde_2.get_state()

        sde_2_state.save(os.path.join(tmpdir, f"state_{i}.h5"))

        sde_2_state = data.SpatialDifferentialExpressionSamplerState.read_h5(
            os.path.join(tmpdir, f"state_{i}.h5")
        )
        sde_2 = spatial_expression.SpatialDifferentialExpression.load_from_state(
            sde_2_state
        )

        sde_2_outputs.append(
            sde_2.sample(
                copy.deepcopy(n_obs_vector),
                copy.deepcopy(Y_igk),
                copy.deepcopy(cell_type_filter),
            )
        )

    for samples_1, samples_2 in zip(sde_1_outputs, sde_2_outputs):
        for a, b in zip(samples_1, samples_2):
            numpy.testing.assert_equal(a, b)

    shutil.rmtree(tmpdir)


def test_get_n_cell_correlation():
    result_for_correlated_arrays = spatial_expression.get_n_cell_correlation(
        np.array([1, 2, 3, 4, 5]), np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    )
    np.random.seed(0)
    result_for_random = spatial_expression.get_n_cell_correlation(
        np.random.random(5), np.random.random(5)
    )

    assert result_for_correlated_arrays > result_for_random


def test_morans_i():
    square_dim = 9
    size = int(square_dim**2)

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
    half_size = int((square_dim**2) / 2)
    half_size_2 = size - half_size
    clustered_w_pattern = np.concatenate([np.ones(half_size), np.zeros(half_size_2)])

    positions = np.zeros((size, 2))

    positions[:, 1] = np.concatenate([np.arange(square_dim)] * square_dim)
    positions[:, 0] = np.array(
        np.concatenate([np.array([i] * square_dim) for i in range(square_dim)])
    )

    positions = positions.astype(int)

    edges = utils.get_edges(positions, layout=data.Layout.SQUARE.value)

    clustered_value = spatial_expression.moran_i(edges, clustered_w_pattern)
    dispersed_value = spatial_expression.moran_i(edges, checkerboard_w_pattern)
    random_value = spatial_expression.moran_i(edges, random_w_pattern)

    assert dispersed_value < random_value < clustered_value
    assert dispersed_value < 0
    assert clustered_value > 0


def test_filter_disconnected_points():
    edges, data = spatial_expression.filter_disconnected_points(
        np.array([[1, 2], [4, 5]]), np.array([1, 2, 3, 4, 5, 6, 7])
    )

    np.testing.assert_array_equal(data, np.array([2, 3, 5, 6]))
    np.testing.assert_array_equal(edges, np.array([[0, 1], [2, 3]]))


def test_morans_i_doesnt_fail_on_disconnected_points():
    data = np.array([1, 2, 3, 4])

    edges = np.array([[1, 2], [2, 3]])
    spatial_expression.moran_i(edges, data)


def test_select_significant_spatial_programs():
    n_genes = 3
    n_components = 1
    n_samples = 10
    n_spatial_patterns = 1

    dataset = bayestme.synthetic_data.generate_fake_stdataset(
        n_rows=50, n_cols=50, n_genes=n_genes
    )

    deconvolution_results = generate_fake_deconvolve_results(
        n_samples=n_samples,
        n_tissue_spots=dataset.n_spot_in,
        n_components=n_components,
        n_genes=n_genes,
    )

    sde_results = generate_fake_sde_results(
        n_samples=n_samples,
        n_genes=n_genes,
        n_components=n_components,
        n_spatial_patterns=n_spatial_patterns,
        n_spot_in=dataset.n_spot_in,
    )

    mock_setups = [
        [
            0.99,
            np.array([0.99, 0.99, 0.99]),
            np.array([0, 1, 2]),
            0.99,
        ],  # Drop programs
        [0.1, np.array([0.1, 0.1, 0.1]), np.array([0, 1, 2]), 0.99],  # Drop programs
        [0.1, np.array([0.99, 0.99, 0.99]), np.array([]), 0.99],  # Drop programs
        [0.1, np.array([0.99, 0.99, 0.99]), np.array([0, 1, 2]), 0.01],  # Drop programs
        [
            0.1,
            np.array([0.99, 0.99, 0.99]),
            np.array([0, 1, 2]),
            0.99,
        ],  # Accept programs
    ]

    expected_results = [[], [], [], [], [(0, 1, np.array([0, 1, 2]))]]

    for mock_setup, expected_result in zip(mock_setups, expected_results):
        with mock.patch(
            "bayestme.spatial_expression.get_n_cell_correlation"
        ) as mock_get_n_cell_correlation:
            with mock.patch(
                "bayestme.spatial_expression.get_proportion_of_spots_in_k_with_pattern_h_per_gene"
            ) as mock_get_proportion_of_spots_in_k_with_pattern_h_per_gene:
                with mock.patch(
                    "bayestme.spatial_expression.filter_pseudogenes_from_selection"
                ) as mock_filter_pseudogenes_from_selection:
                    with mock.patch(
                        "bayestme.spatial_expression.moran_i"
                    ) as mock_moran_i:
                        (
                            cell_correlation,
                            proportions,
                            pseudogene_filter,
                            morans_i,
                        ) = mock_setup

                        mock_get_n_cell_correlation.return_value = cell_correlation
                        mock_get_proportion_of_spots_in_k_with_pattern_h_per_gene.return_value = (
                            proportions
                        )

                        mock_filter_pseudogenes_from_selection.return_value = (
                            pseudogene_filter
                        )

                        mock_moran_i.return_value = morans_i

                        significant_spatial_programs = [
                            _
                            for _ in spatial_expression.select_significant_spatial_programs(
                                stdata=dataset,
                                decon_result=deconvolution_results,
                                sde_result=sde_results,
                                tissue_threshold=1,
                                cell_correlation_threshold=0.5,
                                moran_i_score_threshold=0.9,
                                gene_spatial_pattern_proportion_threshold=0.95,
                                filter_pseudogenes=True,
                            )
                        ]
                        assert len(significant_spatial_programs) == len(expected_result)
                        for i, significant_spatial_program in enumerate(
                            significant_spatial_programs
                        ):
                            assert (
                                significant_spatial_programs[i][0]
                                == expected_result[i][0]
                            )
                            assert (
                                significant_spatial_programs[i][1]
                                == expected_result[i][1]
                            )
                            testing.assert_equal(
                                significant_spatial_programs[i][2],
                                expected_result[i][2],
                            )


def test_plot_spatial_pattern_with_legend():
    np.random.seed(100)
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
        n_rows=50, n_cols=50, n_genes=n_genes
    )

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(
            [
                "looong name",
                "big big name",
                "eirbgoewqugberf:erferf",
                "304ofh308fh3wf:sdfsdfsdr",
                "erferfserf:44",
                "fsdrfsdrgdsrv98dvfj",
                "f34fawefc",
            ]
        ),
        layout=data.Layout.SQUARE,
    )
    deconvolution_results = generate_fake_deconvolve_results(
        n_samples=n_samples,
        n_tissue_spots=dataset.n_spot_in,
        n_components=n_components,
        n_genes=n_genes,
    )

    sde_results = generate_fake_sde_results(
        n_samples=n_samples,
        n_genes=n_genes,
        n_components=n_components,
        n_spatial_patterns=n_spatial_patterns,
        n_spot_in=dataset.n_spot_in,
    )

    tempdir = tempfile.mkdtemp()

    try:
        spatial_expression.plot_spatial_pattern_with_legend(
            stdata=dataset,
            decon_result=deconvolution_results,
            sde_result=sde_results,
            k=1,
            h=1,
            gene_ids=np.array([0, 1, 2]),
            output_file=os.path.join(tempdir, "test.pdf"),
        )
    finally:
        shutil.rmtree(tempdir)


def test_plot_spatial_pattern_and_all_constituent_genes():
    np.random.seed(100)
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
        n_rows=50, n_cols=50, n_genes=n_genes
    )

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(
            [
                "looong name",
                "big big name",
                "eirbgoewqugberf:erferf",
                "304ofh308fh3wf:sdfsdfsdr",
                "erferfserf:44",
                "fsdrfsdrgdsrv98dvfj",
                "f34fawefc",
            ]
        ),
        layout=data.Layout.SQUARE,
    )
    deconvolution_results = generate_fake_deconvolve_results(
        n_samples=n_samples,
        n_tissue_spots=dataset.n_spot_in,
        n_components=n_components,
        n_genes=n_genes,
    )

    sde_results = generate_fake_sde_results(
        n_samples=n_samples,
        n_genes=n_genes,
        n_components=n_components,
        n_spatial_patterns=n_spatial_patterns,
        n_spot_in=dataset.n_spot_in,
    )

    tempdir = tempfile.mkdtemp()

    try:
        for cell_type_name in [None, "name"]:
            spatial_expression.plot_spatial_pattern_and_all_constituent_genes(
                stdata=dataset,
                decon_result=deconvolution_results,
                sde_result=sde_results,
                k=1,
                h=1,
                program_id=1,
                gene_ids=np.array([0, 1, 2]),
                output_dir=tempdir,
                output_format="pdf",
                cell_type_name=cell_type_name,
            )
    finally:
        shutil.rmtree(tempdir)


def test_plot_significant_spatial_patterns():
    with mock.patch(
        "bayestme.spatial_expression.select_significant_spatial_programs"
    ) as mock_select_significant_spatial_programs:
        mock_select_significant_spatial_programs.return_value = [
            (1, 1, [1, 2, 3]),
            (1, 7, [4, 5, 6]),
        ]

        with mock.patch(
            "bayestme.spatial_expression.plot_spatial_pattern_and_all_constituent_genes"
        ) as mock_plot_spatial_pattern_and_all_constituent_genes:
            stdata = mock.MagicMock()
            decon_result = mock.MagicMock()
            sde_result = mock.MagicMock()

            spatial_expression.plot_significant_spatial_patterns(
                stdata=stdata,
                decon_result=decon_result,
                sde_result=sde_result,
                output_dir="output_dir",
                output_format="pdf",
            )

            mock_plot_spatial_pattern_and_all_constituent_genes.assert_has_calls(
                [
                    mock.call(
                        stdata=stdata,
                        decon_result=decon_result,
                        sde_result=sde_result,
                        gene_ids=[1, 2, 3],
                        k=1,
                        h=1,
                        program_id=1,
                        output_dir="output_dir",
                        output_format="pdf",
                        cell_type_name=None,
                    ),
                    mock.call(
                        stdata=stdata,
                        decon_result=decon_result,
                        sde_result=sde_result,
                        gene_ids=[4, 5, 6],
                        k=1,
                        h=7,
                        program_id=2,
                        output_dir="output_dir",
                        output_format="pdf",
                        cell_type_name=None,
                    ),
                ],
                any_order=True,
            )


def test_plot_significant_spatial_patterns_with_cell_type_names():
    with mock.patch(
        "bayestme.spatial_expression.select_significant_spatial_programs"
    ) as mock_select_significant_spatial_programs:
        mock_select_significant_spatial_programs.return_value = [
            (0, 1, [1, 2, 3]),
            (1, 7, [4, 5, 6]),
        ]

        with mock.patch(
            "bayestme.spatial_expression.plot_spatial_pattern_and_all_constituent_genes"
        ) as mock_plot_spatial_pattern_and_all_constituent_genes:
            stdata = mock.MagicMock()
            decon_result = mock.MagicMock()
            sde_result = mock.MagicMock()

            spatial_expression.plot_significant_spatial_patterns(
                stdata=stdata,
                decon_result=decon_result,
                sde_result=sde_result,
                output_dir="output_dir",
                output_format="pdf",
                cell_type_names=["type1", "type2"],
            )

            mock_plot_spatial_pattern_and_all_constituent_genes.assert_has_calls(
                [
                    mock.call(
                        stdata=stdata,
                        decon_result=decon_result,
                        sde_result=sde_result,
                        gene_ids=[1, 2, 3],
                        k=0,
                        h=1,
                        program_id=1,
                        output_dir="output_dir",
                        output_format="pdf",
                        cell_type_name="type1",
                    ),
                    mock.call(
                        stdata=stdata,
                        decon_result=decon_result,
                        sde_result=sde_result,
                        gene_ids=[4, 5, 6],
                        k=1,
                        h=7,
                        program_id=1,
                        output_dir="output_dir",
                        output_format="pdf",
                        cell_type_name="type2",
                    ),
                ],
                any_order=True,
            )
