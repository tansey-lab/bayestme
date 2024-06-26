import os
import shutil
import tempfile
from unittest import mock

import numpy as np

import bayestme
import bayestme.common
import bayestme.data
import bayestme.synthetic_data
import bayestme.utils
from bayestme import data
from bayestme.cli import select_marker_genes


def test_select_marker_genes():
    np.random.seed(101)
    n_genes = 50
    n_marker_genes = 5
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12, n_cols=12, n_genes=n_genes
    )

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(["gene{}".format(x) for x in range(n_genes)]),
        layout=bayestme.common.Layout.SQUARE,
        edges=bayestme.utils.get_edges(locations, bayestme.common.Layout.SQUARE),
    )

    deconvolve_results = bayestme.synthetic_data.create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in, n_components=5, n_samples=100, n_gene=dataset.n_gene
    )

    bayestme.data.add_deconvolution_results_to_dataset(
        stdata=dataset, result=deconvolve_results
    )

    tmpdir = tempfile.mkdtemp()

    stdata_fn = os.path.join(tmpdir, "data.h5")
    adata_output_fn = os.path.join(tmpdir, "data_out.h5")
    deconvolve_results_fn = os.path.join(tmpdir, "deconvolve.h5")
    dataset.save(stdata_fn)
    deconvolve_results.save(deconvolve_results_fn)

    command_line_arguments = [
        "select_marker_genes",
        "--adata",
        stdata_fn,
        "--adata-output",
        adata_output_fn,
        "--deconvolution-result",
        deconvolve_results_fn,
        "--n-marker-genes",
        str(n_marker_genes),
    ]

    try:
        with mock.patch("sys.argv", command_line_arguments):
            select_marker_genes.main()

            stdata = data.SpatialExpressionDataset.read_h5(adata_output_fn)

            assert stdata.marker_gene_indices is not None
            assert stdata.marker_gene_names is not None
    finally:
        shutil.rmtree(tmpdir)
