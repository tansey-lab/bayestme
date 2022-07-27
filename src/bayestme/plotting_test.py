import shutil
import tempfile
import os
import numpy as np

from bayestme import plotting, synthetic_data, data

from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm


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


def test_repro_error():
    hex_stdata = synthetic_data.generate_fake_stdataset(n_genes=1, layout=data.Layout.HEX)

    fig, ax = plt.subplots()

    gene_idx = 0

    counts = hex_stdata.raw_counts[:, gene_idx]
    counts = counts
    positions = hex_stdata.positions

    counts = np.random.uniform(low=-0.15, high=2, size = counts.shape[0])
    plot_mask = np.random.random(size=counts.shape[0]) < 0.25

    counts[~plot_mask] = 0

    vmin = min(-1e-4, counts[plot_mask].min())
    vmax = max(1e-4, counts[plot_mask].max())
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plotting.plot_colored_spatial_polygon(
        fig=fig,
        ax=ax,
        coords=positions,
        values=counts,
        layout=hex_stdata.layout,
    norm=norm,
    colormap=cm.coolwarm)
    plt.show()

    assert 1 == 1

