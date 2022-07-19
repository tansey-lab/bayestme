import shutil
import numpy as np
import tempfile
import os

import bayestme.synthetic_data
from bayestme import bleeding_correction, data


def test_calculate_pairwise_coordinate_differences():
    result = bleeding_correction.calculate_pairwise_coordinate_differences(
        np.array([[0, 0], [1, 1], [2, 2]])
    )
    expected = np.array([[[0, 0],
                          [1, 1],
                          [2, 2]],

                         [[-1, -1],
                          [0, 0],
                          [1, 1]],

                         [[-2, -2],
                          [-1, -1],
                          [0, 0]]])

    np.testing.assert_equal(result, expected)


def test_build_basis_indices():
    basis_idxs_observed, basis_mask_observed = bleeding_correction.build_basis_indices(
        np.array([[0, 0], [0, 1], [0, 3]]), np.array([False, True, False]))

    basis_idxs_expected = np.array([[[0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 0, 0],
                                     [0, 0, 2, 0, 0, 0, 1, 0]],

                                    [[0, 0, 0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 1, 0]],

                                    [[0, 0, 0, 2, 0, 0, 0, 1],
                                     [0, 0, 0, 1, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0, 0, 0, 0]]])

    basis_mask_expected = np.array([[[False, False, False, False, False, False, False, False],
                                     [True, False, True, False, True, False, True, False],
                                     [True, False, True, False, True, False, True, False]],

                                    [[True, False, False, True, True, False, False, True],
                                     [False, False, False, False, False, False, False, False],
                                     [True, False, True, False, True, False, True, False]],

                                    [[True, False, False, True, True, False, False, True],
                                     [True, False, False, True, True, False, False, True],
                                     [False, False, False, False, False, False, False, False]]])

    np.testing.assert_equal(basis_idxs_observed, basis_idxs_expected)
    np.testing.assert_equal(basis_mask_observed, basis_mask_expected)


def test_decontaminate_spots():
    np.random.seed(100)
    # This is just a smoke test, eventually we should make some assertions here
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=3,
        n_cols=3,
        n_genes=1)

    basis_idx, basis_mask = bleeding_correction.build_basis_indices(locations, tissue_mask)

    global_rates, rates, basis_functions, weights, basis_init, rates_init = bleeding_correction.decontaminate_spots(
        Reads=bleed_counts,
        tissue_mask=tissue_mask,
        basis_idxs=basis_idx,
        basis_mask=basis_mask)


def test_select_local_weight():
    np.random.seed(100)
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12,
        n_cols=12,
        n_genes=1)

    basis_indices, basis_mask = bleeding_correction.build_basis_indices(locations, tissue_mask)

    best_local_weight, delta_local_weight, lw_losses, lw_grid, (
    best_basis_init, best_rates_init) = bleeding_correction.select_local_weight(
        bleed_counts, tissue_mask, basis_indices, basis_mask, n_weights=3)


def test_fit_basis_functions():
    np.random.seed(100)
    # This is just a smoke test, eventually we should make some assertions here
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=9,
        n_cols=9,
        n_genes=1)

    basis_idxs, basis_mask = bleeding_correction.build_basis_indices(locations, tissue_mask)
    rates = np.copy(bleed_counts) * tissue_mask[:, None] * bleeding_correction.RATE_INITIALIZATION_FACTOR
    global_rates = np.median(bleed_counts[:, :1], axis=0)
    bleeding_correction.fit_basis_functions(
        bleed_counts,
        tissue_mask,
        rates,
        global_rates,
        basis_idxs,
        basis_mask)


def test_multinomial():
    import torch
    from torch.distributions.multinomial import Multinomial
    Multinomial(total_count=10, probs=torch.tensor([1, 1, 1]))


def test_clean_bleed():
    np.random.seed(100)
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12,
        n_cols=12,
        n_genes=5)

    dataset = data.SpatialExpressionDataset(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(['1', '2', '3', '4', '5']),
        layout=data.Layout.SQUARE
    )

    (cleaned_dataset, bleed_correction_result) = bleeding_correction.clean_bleed(
        dataset,
        n_top=3,
        local_weight=None)

    assert cleaned_dataset.n_gene == 5
    assert bleed_correction_result.corrected_reads.shape == (12*12, 5)
    assert bleed_correction_result.global_rates.shape == (5, )


def test_plot_bleed_vectors():
    np.random.seed(100)
    dataset = bayestme.synthetic_data.generate_fake_stdataset(12,
                                                    12,
                                                    2,
                                                    data.Layout.SQUARE)

    (cleaned_dataset, bleed_correction_result) = bleeding_correction.clean_bleed(
        dataset,
        n_top=3,
        local_weight=None)

    tempdir = tempfile.mkdtemp()
    try:
        bleeding_correction.plot_bleed_vectors(
            stdata=cleaned_dataset,
            bleed_result=bleed_correction_result,
            gene_name='1',
            output_path=os.path.join(tempdir, 'bleed_plot.pdf')
        )
    finally:
        shutil.rmtree(tempdir)


def test_clean_bleed_plots():
    np.random.seed(100)
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12,
        n_cols=12,
        n_genes=5)

    dataset = data.SpatialExpressionDataset(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(['1', '2', '3', '4', '5']),
        layout=data.Layout.SQUARE
    )

    (cleaned_dataset, bleed_correction_result) = bleeding_correction.clean_bleed(
        dataset,
        n_top=3,
        local_weight=None)

    tempdir = tempfile.mkdtemp()
    try:
        bleeding_correction.plot_basis_functions(
            basis_functions=bleed_correction_result.basis_functions,
            output_dir=tempdir
        )

        for gene in ['1', '2', '3', '4', '5']:
            bleeding_correction.plot_bleed_vectors(
                stdata=cleaned_dataset,
                bleed_result=bleed_correction_result,
                gene_name=gene,
                output_path=os.path.join(tempdir, 'bleed_plot.pdf')
            )
            bleeding_correction.plot_bleeding(
                before_correction=dataset,
                after_correction=cleaned_dataset,
                gene=gene,
                output_path=os.path.join(tempdir, 'bleeding.png'))
            bleeding_correction.plot_before_after_cleanup(
                before_correction=dataset, after_correction=cleaned_dataset, gene=gene, output_dir=tempdir)
    finally:
        shutil.rmtree(tempdir)


def test_create_top_n_gene_bleeding_plots():
    np.random.seed(100)
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12,
        n_cols=12,
        n_genes=5)

    dataset = data.SpatialExpressionDataset(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(['1', '2', '3', '4', '5']),
        layout=data.Layout.SQUARE
    )

    (cleaned_dataset, bleed_correction_result) = bleeding_correction.clean_bleed(
        dataset,
        n_top=3,
        local_weight=None)

    tempdir = tempfile.mkdtemp()
    try:
        bleeding_correction.create_top_n_gene_bleeding_plots(
            dataset=dataset,
            corrected_dataset=cleaned_dataset,
            bleed_result=bleed_correction_result,
            output_dir=tempdir,
            n_genes=3
        )
    finally:
        shutil.rmtree(tempdir)
