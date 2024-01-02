import shutil
import tempfile
import os.path
import numpy as np

import bayestme.common
import bayestme.expression_truth
import bayestme.synthetic_data
from bayestme import data
from bayestme.svi import deconvolution


def test_deconvolve_with_no_spatial_guide():
    n_genes = 5
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=15, n_cols=15, n_genes=n_genes
    )

    stdata = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(["{}".format(x) for x in range(n_genes)]),
        layout=bayestme.common.Layout.SQUARE,
        barcodes=np.array(["barcode" + str(i) for i in range(len(locations))]),
    )
    K = 3
    n_traces = 7

    rng = np.random.default_rng(42)

    result = bayestme.svi.deconvolution.deconvolve(
        stdata=stdata,
        n_components=K,
        rho=0.5,
        n_svi_steps=10,
        n_samples=n_traces,
        use_spatial_guide=False,
        rng=rng,
    )

    assert result.beta_trace.shape == (
        n_traces,
        K,
    )
    assert result.expression_trace.shape == (n_traces, K, n_genes)
    assert result.cell_prob_trace.shape == (n_traces, stdata.n_spot_in, K)
    assert result.cell_num_trace.shape == (n_traces, stdata.n_spot_in, K)
    assert result.reads_trace.shape == (n_traces, stdata.n_spot_in, n_genes, K)


def test_deconvolve_with_no_spatial_guide_and_expression_truth():
    n_genes = 5
    K = 3
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=15, n_cols=15, n_genes=n_genes
    )

    expression_truth = np.random.poisson(1, (K, n_genes)) + 0.1

    stdata = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(["{}".format(x) for x in range(n_genes)]),
        layout=bayestme.common.Layout.SQUARE,
        barcodes=np.array(["barcode" + str(i) for i in range(len(locations))]),
    )

    K = 3
    n_traces = 7

    rng = np.random.default_rng(42)

    result = bayestme.svi.deconvolution.deconvolve(
        stdata=stdata,
        n_components=K,
        rho=0.5,
        n_svi_steps=10,
        n_samples=n_traces,
        use_spatial_guide=False,
        expression_truth=expression_truth,
        rng=rng,
    )

    assert result.beta_trace.shape == (
        n_traces,
        K,
    )
    assert result.expression_trace.shape == (n_traces, K, n_genes)
    assert result.cell_prob_trace.shape == (n_traces, stdata.n_spot_in, K)
    assert result.cell_num_trace.shape == (n_traces, stdata.n_spot_in, K)
    assert result.reads_trace.shape == (n_traces, stdata.n_spot_in, n_genes, K)


def test_deconvolve_with_spatial_guide():
    n_genes = 5
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=15, n_cols=15, n_genes=n_genes
    )

    stdata = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(["{}".format(x) for x in range(n_genes)]),
        layout=bayestme.common.Layout.SQUARE,
        barcodes=np.array(["barcode" + str(i) for i in range(len(locations))]),
    )

    K = 3
    n_traces = 7

    rng = np.random.default_rng(42)

    result = bayestme.svi.deconvolution.deconvolve(
        stdata=stdata,
        n_components=K,
        rho=0.5,
        n_svi_steps=10,
        n_samples=n_traces,
        use_spatial_guide=True,
        expression_truth=None,
        rng=rng,
    )

    assert result.beta_trace.shape == (
        n_traces,
        K,
    )
    assert result.expression_trace.shape == (n_traces, K, n_genes)
    assert result.cell_prob_trace.shape == (n_traces, stdata.n_spot_in, K)
    assert result.cell_num_trace.shape == (n_traces, stdata.n_spot_in, K)
    assert result.reads_trace.shape == (n_traces, stdata.n_spot_in, n_genes, K)


def test_deconvolve_with_spatial_guide_and_expression_truth():
    n_genes = 5
    K = 3
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=15, n_cols=15, n_genes=n_genes
    )

    expression_truth = np.random.poisson(1, (K, n_genes)) + 0.1

    stdata = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(["{}".format(x) for x in range(n_genes)]),
        layout=bayestme.common.Layout.SQUARE,
        barcodes=np.array(["barcode" + str(i) for i in range(len(locations))]),
    )

    n_traces = 7

    rng = np.random.default_rng(42)

    result = bayestme.svi.deconvolution.deconvolve(
        stdata=stdata,
        n_components=K,
        rho=0.5,
        n_svi_steps=10,
        n_samples=n_traces,
        use_spatial_guide=True,
        expression_truth=expression_truth,
        rng=rng,
    )

    assert result.beta_trace.shape == (
        n_traces,
        K,
    )
    assert result.expression_trace.shape == (n_traces, K, n_genes)
    assert result.cell_prob_trace.shape == (n_traces, stdata.n_spot_in, K)
    assert result.cell_num_trace.shape == (n_traces, stdata.n_spot_in, K)
    assert result.reads_trace.shape == (n_traces, stdata.n_spot_in, n_genes, K)
