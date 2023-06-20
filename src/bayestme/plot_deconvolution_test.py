import shutil
import tempfile

import bayestme.plot_deconvolution
from bayestme.mcmc.deconvolution_test import create_deconvolve_dataset


def test_deconvolve_plots():
    tempdir = tempfile.mkdtemp()

    dataset = create_deconvolve_dataset()

    try:
        bayestme.plot_deconvolution.plot_deconvolution(
            stdata=dataset, output_dir=tempdir
        )
    finally:
        shutil.rmtree(tempdir)


def test_deconvolve_plots_with_cell_type_names():
    tempdir = tempfile.mkdtemp()
    dataset = create_deconvolve_dataset(n_components=5)
    try:
        bayestme.plot_deconvolution.plot_deconvolution(
            stdata=dataset,
            output_dir=tempdir,
            cell_type_names=["type1", "banana", "threeve", "quattro", "ISPC"],
        )
    finally:
        shutil.rmtree(tempdir)
