import os
import shutil
import tempfile
from unittest import mock

import bayestme.synthetic_data
from bayestme import data
from bayestme.cli import spatial_transcriptional_programs
from bayestme.data_test import generate_toy_stdataset


def test_stp():
    dataset = generate_toy_stdataset()
    tmpdir = tempfile.mkdtemp()

    input_path = os.path.join(tmpdir, "data.h5")
    deconvolve_path = os.path.join(tmpdir, "deconvolve.h5")
    stp_path = os.path.join(tmpdir, "stp.h5")

    deconvolve_rv = bayestme.synthetic_data.create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in, n_components=2, n_samples=100, n_gene=dataset.n_gene
    )

    command_line_arguments = [
        "spatial_transcriptional_programs",
        "--adata",
        input_path,
        "--deconvolution-result",
        deconvolve_path,
        "--output",
        stp_path,
        "--trend-filtering-lambda",
        "0.1",
        "--lasso-lambda",
        "0.1",
        "--n-spatial-programs",
        "2",
        "--n-iter",
        "3",
        "--seed",
        "42",
    ]

    try:
        dataset.save(input_path)

        deconvolve_rv.save(deconvolve_path)

        with mock.patch("sys.argv", command_line_arguments):
            spatial_transcriptional_programs.main()
        result = data.SpatialDifferentialExpressionResult.read_h5(stp_path)

    finally:
        shutil.rmtree(tmpdir)
