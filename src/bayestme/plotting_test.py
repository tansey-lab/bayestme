import shutil
import tempfile
import os

from bayestme import plotting, synthetic_data, data

from matplotlib import pyplot as plt


def test_plot_gene_raw_counts():
    sq_stdata = synthetic_data.generate_fake_stdataset(n_genes=1, layout=data.Layout.SQUARE)
    hex_stdata = synthetic_data.generate_fake_stdataset(n_genes=1, layout=data.Layout.HEX)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    gene_idx = 0

    counts = sq_stdata.raw_counts[:, gene_idx]
    counts = counts
    positions = sq_stdata.positions

    plotting.plot_colored_spatial_polygon(
        fig=fig,
        ax=ax1,
        coords=positions,
        values=counts,
        layout=sq_stdata.layout)

    counts = hex_stdata.raw_counts[:, gene_idx]
    counts = counts
    positions = hex_stdata.positions

    plotting.plot_colored_spatial_polygon(
        fig=fig,
        ax=ax2,
        coords=positions,
        values=counts,
        layout=hex_stdata.layout)

    tempdir = tempfile.mkdtemp()
    try:
        fig.savefig(os.path.join(tempdir, 'test.pdf'))
    finally:
        shutil.rmtree(tempdir)


