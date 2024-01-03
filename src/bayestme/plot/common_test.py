import os
import shutil
import tempfile
import numpy as np
from matplotlib import pyplot as plt

import bayestme.common
import bayestme.plot.common
from bayestme import synthetic_data


def test_plot_gene_raw_counts():
    sq_stdata = synthetic_data.generate_fake_stdataset(
        n_genes=1, layout=bayestme.common.Layout.SQUARE
    )
    hex_stdata = synthetic_data.generate_fake_stdataset(
        n_genes=1, layout=bayestme.common.Layout.HEX
    )
    irreg_stdata = synthetic_data.generate_fake_stdataset(
        n_genes=1, layout=bayestme.common.Layout.IRREGULAR
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

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

    counts = irreg_stdata.raw_counts[:, gene_idx]
    counts = counts
    positions = irreg_stdata.positions
    bayestme.plot.common.plot_colored_spatial_polygon(
        fig=fig, ax=ax3, coords=positions, values=counts, layout=irreg_stdata.layout
    )

    tempdir = tempfile.mkdtemp()
    try:
        fig.savefig(os.path.join(tempdir, "test.pdf"))
    finally:
        shutil.rmtree(tempdir)


def test_plot_scatterpie():
    sq_stdata = synthetic_data.generate_fake_stdataset(
        n_genes=1, layout=bayestme.common.Layout.SQUARE
    )
    hex_stdata = synthetic_data.generate_fake_stdataset(
        n_genes=1, layout=bayestme.common.Layout.HEX
    )
    irreg_stdata = synthetic_data.generate_fake_stdataset(
        n_genes=1, layout=bayestme.common.Layout.IRREGULAR
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    values = np.random.random((sq_stdata.n_spot_in, 5))
    positions = sq_stdata.positions_tissue

    bayestme.plot.common.plot_spatial_pie_charts(
        fig=fig, ax=ax1, coords=positions, values=values, layout=sq_stdata.layout
    )

    values = np.random.random((hex_stdata.n_spot_in, 5))
    positions = hex_stdata.positions_tissue

    bayestme.plot.common.plot_spatial_pie_charts(
        fig=fig, ax=ax2, coords=positions, values=values, layout=hex_stdata.layout
    )

    values = np.random.random((irreg_stdata.n_spot_in, 5))
    positions = irreg_stdata.positions_tissue
    bayestme.plot.common.plot_spatial_pie_charts(
        fig=fig, ax=ax3, coords=positions, values=values, layout=irreg_stdata.layout
    )

    tempdir = tempfile.mkdtemp()
    try:
        fig.savefig(os.path.join(tempdir, "test.pdf"))
    finally:
        shutil.rmtree(tempdir)
