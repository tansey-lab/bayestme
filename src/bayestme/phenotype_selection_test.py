import shutil
import tempfile

import numpy as np
import pytest

import bayestme.utils
import bayestme.common
import bayestme.synthetic_data
from bayestme import data, phenotype_selection
from bayestme.common import InferenceType


def test_get_phenotype_selection_parameters_for_folds():
    n_genes = 3
    n_components_min = 2
    n_components_max = 3
    n_fold = 2
    n_splits = 15
    lams = [1, 10]
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=50, n_cols=50, n_genes=n_genes
    )

    stdata = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(["gene{}".format(x) for x in range(n_genes)]),
        layout=bayestme.common.Layout.SQUARE,
        edges=bayestme.utils.get_edges(locations, bayestme.common.Layout.SQUARE),
    )

    g = phenotype_selection.get_phenotype_selection_parameters_for_folds(
        stdata=stdata,
        n_fold=n_fold,
        n_splits=n_splits,
        lams=lams,
        n_components_max=n_components_max,
        n_components_min=n_components_min,
    )

    folds = [_ for _ in g]

    assert len(folds) == (
        ((n_components_max - n_components_min) + 1) * n_fold * len(lams)
    )


def test_plot_folds():
    n_genes = 3
    n_components_min = 2
    n_components_max = 3
    n_fold = 2
    n_splits = 15
    lams = [1, 10]
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=50, n_cols=50, n_genes=n_genes
    )

    stdata = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(["gene{}".format(x) for x in range(n_genes)]),
        layout=bayestme.common.Layout.SQUARE,
        edges=bayestme.utils.get_edges(locations, bayestme.common.Layout.SQUARE),
    )

    g = phenotype_selection.create_folds(
        stdata=stdata, n_fold=n_fold, n_splits=n_splits
    )

    folds = [_ for _ in g]

    tempdir = tempfile.mkdtemp()

    try:
        phenotype_selection.plot_folds(stdata, folds, tempdir)
    finally:
        shutil.rmtree(tempdir)


@pytest.mark.parametrize("inference_type", [InferenceType.MCMC, InferenceType.SVI])
def test_run_phenotype_selection_single_fold(inference_type):
    n_genes = 10
    n_samples = 2
    n_components = 3
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12, n_cols=12, n_genes=n_genes
    )

    stdata = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(["gene{}".format(x) for x in range(n_genes)]),
        layout=bayestme.common.Layout.SQUARE,
        edges=bayestme.utils.get_edges(locations, bayestme.common.Layout.SQUARE),
    )

    params = [
        _
        for _ in phenotype_selection.get_phenotype_selection_parameters_for_folds(
            stdata,
            n_fold=2,
            n_splits=2,
            lams=[10],
            n_components_min=n_components,
            n_components_max=n_components,
        )
    ]

    lam, n_components_for_job, mask, fold_number = params[0]

    result = phenotype_selection.run_phenotype_selection_single_job(
        spatial_smoothing_parameter=lam,
        n_components=n_components_for_job,
        mask=mask,
        fold_number=fold_number,
        stdata=stdata,
        n_samples=n_samples,
        mcmc_n_burn=1,
        mcmc_n_thin=1,
        n_svi_steps=1,
        background_noise=False,
        lda_initialization=False,
        use_spatial_guide=False,
        inference_type=inference_type,
    )

    assert result.cell_prob_trace.shape == (
        n_samples,
        stdata.n_spot_in,
        n_components,
    )
    assert result.cell_num_trace.shape == (
        n_samples,
        stdata.n_spot_in,
        n_components,
    )
    assert result.expression_trace.shape == (n_samples, n_components, n_genes)
    assert result.beta_trace.shape == (n_samples, n_components)
    assert result.log_lh_test_trace.shape == (n_samples,)
    assert result.log_lh_train_trace.shape == (n_samples,)
