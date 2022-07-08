import shutil

import numpy as np
import tempfile

from bayestme import bleeding_correction, data, phenotype_selection


def test_get_phenotype_selection_parameters_for_folds():
    n_genes = 3
    n_components_min = 2
    n_components_max = 3
    n_fold = 2
    n_splits = 15
    lams = [1, 10]
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bleeding_correction.generate_data(
        n_rows=50,
        n_cols=50,
        n_genes=n_genes)

    stdata = data.SpatialExpressionDataset(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations.T,
        gene_names=np.array(['gene{}'.format(x) for x in range(n_genes)]),
        layout=data.Layout.SQUARE)

    g = phenotype_selection.get_phenotype_selection_parameters_for_folds(
        stdata=stdata,
        n_fold=n_fold,
        n_splits=n_splits,
        lams=lams,
        n_components_max=n_components_max,
        n_components_min=n_components_min,
    )

    folds = [_ for _ in g]

    assert len(folds) == (((n_components_max - n_components_min) + 1) * n_fold * len(lams))


def test_plot_folds():
    n_genes = 3
    n_components_min = 2
    n_components_max = 3
    n_fold = 2
    n_splits = 15
    lams = [1, 10]
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bleeding_correction.generate_data(
        n_rows=50,
        n_cols=50,
        n_genes=n_genes)

    stdata = data.SpatialExpressionDataset(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations.T,
        gene_names=np.array(['gene{}'.format(x) for x in range(n_genes)]),
        layout=data.Layout.SQUARE)

    g = phenotype_selection.get_phenotype_selection_parameters_for_folds(
        stdata=stdata,
        n_fold=n_fold,
        n_splits=n_splits,
        lams=lams,
        n_components_max=n_components_max,
        n_components_min=n_components_min,
    )

    folds = [_ for _ in g]

    tempdir = tempfile.mkdtemp()

    try:
        phenotype_selection.plot_folds(stdata, folds, tempdir)
    finally:
        shutil.rmtree(tempdir)


def test_sample_graph_fused_multinomial():
    n_genes = 50
    n_samples = 3
    n_components = 2
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bleeding_correction.generate_data(
        n_rows=12,
        n_cols=12,
        n_genes=n_genes)

    stdata = data.SpatialExpressionDataset(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations.T,
        gene_names=np.array(['gene{}'.format(x) for x in range(n_genes)]),
        layout=data.Layout.SQUARE)

    mask = np.random.choice(np.array([True, False]), size=stdata.n_spot_in)
    train = stdata.reads.copy()
    test = stdata.reads.copy()
    train[mask] = 0
    test[~mask] = 0

    (
        cell_prob_trace,
        cell_num_trace,
        expression_trace,
        beta_trace,
        loglhtest_trace,
        loglhtrain_trace
    ) = phenotype_selection.sample_graph_fused_multinomial(
        mask=mask,
        train=train,
        test=test,
        n_components=n_components,
        edges=stdata.edges,
        n_gene=n_genes,
        lam_psi=1e-2,
        background_noise=False,
        lda_initialization=False,
        n_max=20,
        n_samples=n_samples,
        n_thin=1,
        n_burn=0)

    assert cell_prob_trace.shape == (n_samples, stdata.n_spot_in, n_components + 1)
    assert cell_num_trace.shape == (n_samples, stdata.n_spot_in, n_components + 1)
    assert expression_trace.shape == (n_samples, n_components, n_genes)
    assert beta_trace.shape == (n_samples, n_components)


def test_run_phenotype_selection_single_fold():
    n_genes = 50
    n_samples = 1
    n_top = 1
    n_components = 3
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bleeding_correction.generate_data(
        n_rows=12,
        n_cols=12,
        n_genes=n_genes)

    stdata = data.SpatialExpressionDataset(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations.T,
        gene_names=np.array(['gene{}'.format(x) for x in range(n_genes)]),
        layout=data.Layout.SQUARE)

    result = phenotype_selection.run_phenotype_selection_single_fold(
        fold_idx=0,
        stdata=stdata,
        n_fold=2,
        n_splits=2,
        lams=[10],
        n_components_min=n_components,
        n_components_max=n_components,
        n_samples=n_samples,
        n_burn=1,
        n_thin=1,
        max_ncell=1,
        n_gene=n_top,
        background_noise=False,
        lda_initialization=False)

    assert result.cell_prob_trace.shape == (n_samples, stdata.n_spot_in, n_components + 1)
    assert result.cell_num_trace.shape == (n_samples, stdata.n_spot_in, n_components + 1)
    assert result.expression_trace.shape == (n_samples, n_components, n_top)
    assert result.beta_trace.shape == (n_samples, n_components)
