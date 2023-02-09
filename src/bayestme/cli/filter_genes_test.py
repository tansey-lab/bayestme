import shutil

import numpy as np
import tempfile
import os
from unittest import mock

from bayestme import data

from bayestme.cli import filter_genes


def test_filter_genes():
    raw_counts = np.array([[199, 200, 1], [10000, 10001, 2], [0, 1, 3]], dtype=np.int64)

    locations = np.array([(x, 0) for x in range(3)])

    tissue_mask = np.array([True for _ in range(3)])

    gene_names = np.array(["keep_me", "filter1", "filter2"])

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=raw_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=gene_names,
        layout=data.Layout.SQUARE,
    )

    tmpdir = tempfile.mkdtemp()

    input_path = os.path.join(tmpdir, "data.h5")
    output_path = os.path.join(tmpdir, "result.h5")

    command_line_arguments = [
        "filter_genes",
        "--adata",
        input_path,
        "--output",
        output_path,
        "--n-top-by-standard-deviation",
        "2",
        "--spot-threshold",
        "0.95",
        "--filter-ribosomal-genes",
    ]

    try:
        dataset.save(os.path.join(tmpdir, "data.h5"))

        with mock.patch("sys.argv", command_line_arguments):
            filter_genes.main()

        result = data.SpatialExpressionDataset.read_h5(output_path)

        np.testing.assert_equal(result.gene_names, np.array([]))
    finally:
        shutil.rmtree(tmpdir)


def test_filter_expression_truth():
    raw_counts = np.array([[199, 200, 1], [10000, 10001, 2], [0, 1, 3]], dtype=np.int64)

    locations = np.array([(x, 0) for x in range(3)])

    tissue_mask = np.array([True for _ in range(3)])

    gene_names = np.array(["keep_me", "filter1", "filter2"])

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=raw_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=gene_names,
        layout=data.Layout.SQUARE,
    )

    tmpdir = tempfile.mkdtemp()

    input_path = os.path.join(tmpdir, "data.h5")
    output_path = os.path.join(tmpdir, "result.h5")

    command_line_arguments = [
        "filter_genes",
        "--adata",
        input_path,
        "--output",
        output_path,
        "--expression-truth",
        "xxx",
    ]

    try:
        dataset.save(os.path.join(tmpdir, "data.h5"))

        with mock.patch("sys.argv", command_line_arguments):
            with mock.patch(
                "bayestme.gene_filtering.filter_stdata_to_match_expression_truth"
            ) as mock_filter_stdata_to_match_expression_truth:
                filter_genes.main()

                mock_filter_stdata_to_match_expression_truth.assert_called_once_with(
                    mock.ANY, "xxx"
                )
    finally:
        shutil.rmtree(tmpdir)
