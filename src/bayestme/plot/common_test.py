import os
import shutil
import tempfile

from matplotlib import pyplot as plt

import bayestme.common
import bayestme.plot.common
from bayestme import synthetic_data, data


def test_plot_gene_raw_counts():
    sq_stdata = synthetic_data.generate_fake_stdataset(
        n_genes=1, layout=bayestme.common.Layout.SQUARE
    )
    hex_stdata = synthetic_data.generate_fake_stdataset(
        n_genes=1, layout=bayestme.common.Layout.HEX
    )

    fig, (ax1, ax2) = plt.subplots(1, 2)

    gene_idx = 0

    counts = sq_stdata.raw_counts[:, gene_idx]
    counts = counts
    positions = sq_stdata.positions

    bayestme.plot.common.plot_colored_spatial_polygon(
        fig=fig, ax=ax1, coords=positions, values=counts, layout=sq_stdata.layout
    )

    counts = hex_stdata.raw_counts[:, gene_idx]
    counts = counts
    positions = hex_stdata.positions

    bayestme.plot.common.plot_colored_spatial_polygon(
        fig=fig, ax=ax2, coords=positions, values=counts, layout=hex_stdata.layout
    )

    tempdir = tempfile.mkdtemp()
    try:
        fig.savefig(os.path.join(tempdir, "test.pdf"))
    finally:
        shutil.rmtree(tempdir)
