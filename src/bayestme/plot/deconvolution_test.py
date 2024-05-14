import shutil
import tempfile

from bayestme.plot import deconvolution
from bayestme.synthetic_data import create_deconvolve_dataset
from bayestme import common


def test_deconvolve_plots():
    tempdir = tempfile.mkdtemp()

    dataset = create_deconvolve_dataset()

    try:
        deconvolution.plot_deconvolution(stdata=dataset, output_dir=tempdir)
    finally:
        shutil.rmtree(tempdir)


def test_deconvolve_plots_with_cell_type_names():
    tempdir = tempfile.mkdtemp()
    dataset = create_deconvolve_dataset(n_components=5)
    try:
        deconvolution.plot_deconvolution(
            stdata=dataset,
            output_dir=tempdir,
            cell_type_names=["type1", "banana", "threeve", "quattro", "ISPC"],
        )
    finally:
        shutil.rmtree(tempdir)

    tempdir = tempfile.mkdtemp()
    dataset = create_deconvolve_dataset(
        n_components=5, n_nodes=20, layout=common.Layout.HEX
    )
    try:
        deconvolution.plot_deconvolution(
            stdata=dataset,
            output_dir=tempdir,
            cell_type_names=["type1", "banana", "threeve", "quattro", "ISPC"],
        )
    finally:
        shutil.rmtree(tempdir)
