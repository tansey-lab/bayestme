import shutil

import numpy as np
import tempfile

from bayestme import spatial_transcriptional_programs
from bayestme.synthetic_data import generate_demo_stp_dataset
from bayestme.svi import deconvolution


def test_stp():
    data = generate_demo_stp_dataset()
    rng = np.random.default_rng(42)

    deconvolve_result = deconvolution.deconvolve(
        stdata=data,
        n_components=2,
        rho=0.5,
        n_svi_steps=3,
        n_samples=10,
        use_spatial_guide=False,
        rng=rng,
    )

    stp_result = spatial_transcriptional_programs.train(
        data=data,
        deconvolution_result=deconvolve_result,
        batchsize_genes=data.n_gene,
        batchsize_spots=data.n_spot_in,
        n_steps=3,
        n_programs=3,
        rng=rng,
    )

    tempdir = tempfile.mkdtemp()

    try:
        spatial_transcriptional_programs.plot_spatial_transcriptional_programs(
            stp_result, data, output_dir=tempdir
        )
    finally:
        shutil.rmtree(tempdir)
