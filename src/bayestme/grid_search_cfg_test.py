from bayestme import bleeding_correction, bayestme_data, grid_search_cfg, utils


def test_sample_graph_fused_multinomial():
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bleeding_correction.generate_data(
        n_rows=15,
        n_cols=15,
        n_genes=3)

    fold_generator = bayestme_data.CrossValidationSTData.create_folds(
        n_spot_in=tissue_mask.sum(),
        positions_tissue=locations.T[:, tissue_mask],
        layout=2,
        reads=bleed_counts[tissue_mask],
        n_fold=5,
        n_splits=5)

    mask, train, test, n_neighbours = next(fold_generator)

    (
        cell_prob_trace,
        cell_num_trace,
        expression_trace,
        beta_trace,
        loglhtest_trace,
        loglhtrain_trace
    ) = grid_search_cfg.sample_graph_fused_multinomial(
        mask=mask,
        train=train,
        test=test,
        n_components=2,
        edges=utils.get_edges(locations.T[:, tissue_mask], 2),
        n_gene=3,
        lam_psi=1e-2,
        background_noise=False,
        lda_initialization=False,
        n_max=20,
        n_samples=1,
        n_thin=1,
        n_burn=0
    )