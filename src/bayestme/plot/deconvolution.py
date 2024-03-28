import os
import numpy as np
from typing import Optional, List

from matplotlib import cm as cm, pyplot as plt, gridspec

from bayestme import data
from bayestme.marker_genes import plot_marker_genes
from bayestme.plot import common


def plot_cell_num(
    stdata: data.SpatialExpressionDataset,
    output_dir: str,
    output_format: str = "pdf",
    cmap=cm.coolwarm,
    seperate_pdf: bool = False,
    cell_type_names: Optional[List[str]] = None,
    n_panels_per_row: int = 3,
):
    plot_object = stdata.cell_type_counts

    if plot_object is None:
        raise RuntimeError("SpatialExpressionDataset contains no cell type counts")

    if seperate_pdf:
        for i in range(stdata.n_cell_types):
            fig, ax = plt.subplot()

            if cell_type_names is not None:
                title = cell_type_names[i]
            else:
                title = f"Cell Type {i + 1}"

            ax.set_title(title)
            common.plot_colored_spatial_polygon(
                fig=fig,
                ax=ax,
                coords=stdata.positions_tissue,
                values=plot_object[:, i],
                layout=stdata.layout,
                colormap=cmap,
            )
            ax.set_axis_off()

            fig.savefig(
                os.path.join(output_dir, f"cell_type_counts_{i}.{output_format}")
            )
            plt.close(fig)
    else:
        n_panels_x = min(n_panels_per_row, stdata.n_cell_types)
        n_panels_y = np.ceil(stdata.n_cell_types / n_panels_x).astype(int)

        fig = plt.figure(
            figsize=(
                n_panels_x * 3,
                n_panels_y * 3,
            )
        )
        gs = gridspec.GridSpec(
            nrows=n_panels_y, ncols=n_panels_x, wspace=0.22, hspace=0.3
        )

        for i in range(stdata.n_cell_types):
            if cell_type_names is not None:
                title = cell_type_names[i]
            else:
                title = f"Cell Type {i + 1}"
            ax = fig.add_subplot(gs[i])

            ax.set_title(title)
            common.plot_colored_spatial_polygon(
                fig=fig,
                ax=ax,
                coords=stdata.positions_tissue,
                values=plot_object[:, i],
                layout=stdata.layout,
                colormap=cmap,
            )
            ax.set_axis_off()
        fig.savefig(os.path.join(output_dir, f"cell_type_counts.{output_format}"))
        plt.close(fig)


def plot_cell_prob(
    stdata: data.SpatialExpressionDataset,
    output_dir: str,
    output_format: str = "pdf",
    cmap=cm.coolwarm,
    seperate_pdf: bool = False,
    cell_type_names: Optional[List[str]] = None,
    n_panels_per_row: int = 3,
):
    plot_object = stdata.cell_type_probabilities
    if plot_object is None:
        raise RuntimeError("SpatialExpressionDataset contains no cell type counts")

    if seperate_pdf:
        for i in range(stdata.n_cell_types):
            fig, ax = plt.subplot()
            if cell_type_names is not None:
                title = cell_type_names[i]
            else:
                title = f"Cell Type {i + 1}"

            ax.set_title(title)
            common.plot_colored_spatial_polygon(
                fig=fig,
                ax=ax,
                coords=stdata.positions_tissue,
                values=plot_object[:, i],
                layout=stdata.layout,
                colormap=cmap,
            )
            ax.set_axis_off()

            fig.savefig(
                os.path.join(output_dir, f"cell_type_probability_{i}.{output_format}")
            )
            plt.close(fig)
    else:
        n_panels_x = min(n_panels_per_row, stdata.n_cell_types)
        n_panels_y = np.ceil(stdata.n_cell_types / n_panels_x).astype(int)

        fig = plt.figure(
            figsize=(
                n_panels_x * 3,
                n_panels_y * 3,
            )
        )
        gs = gridspec.GridSpec(
            nrows=n_panels_y, ncols=n_panels_x, wspace=0.22, hspace=0.3
        )

        for i in range(stdata.n_cell_types):
            if cell_type_names is not None:
                title = cell_type_names[i]
            else:
                title = f"Cell Type {i + 1}"
            ax = fig.add_subplot(gs[i])

            ax.set_title(title)
            common.plot_colored_spatial_polygon(
                fig=fig,
                ax=ax,
                coords=stdata.positions_tissue,
                values=plot_object[:, i],
                layout=stdata.layout,
                colormap=cmap,
            )
            ax.set_axis_off()
        fig.savefig(
            os.path.join(output_dir, f"cell_type_probabilities.{output_format}")
        )
        plt.close(fig)


def plot_cell_num_scatterpie(
    stdata: data.SpatialExpressionDataset,
    output_path: str,
    cell_type_names: Optional[List[str]] = None,
):
    """
    Create a "scatter pie" plot of the deconvolution cell counts.

    :param stdata: SpatialExpressionDataset to plot
    :param output_path: Where to save plot
    :param cell_type_names: Cell type names to use in plot, an array of length n_components
    """
    if stdata.cell_type_counts is None:
        raise RuntimeError("SpatialExpressionDataset contains no cell type counts")

    fig, ax = plt.subplots()

    common.plot_spatial_pie_charts(
        fig,
        ax,
        stdata.positions_tissue,
        values=stdata.cell_type_counts,
        layout=stdata.layout,
        plotting_coordinates=stdata.positions,
        cell_type_names=cell_type_names,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_one_vs_all_cell_num_scatterpie(
    stdata: data.SpatialExpressionDataset,
    output_dir: str,
    cell_type_names: Optional[List[str]] = None,
):
    """
    Create a "scatter pie" plot of the deconvolution cell counts, except
    iterate through each celltype and plot it against all other cell types.
    Will create n_celltypes plots.
    Useful is n_celltypes is large.

    :param stdata: SpatialExpressionDataset to plot
    :param output_dir: Where to save plot
    :param cell_type_names: Cell type names to use in plot, an array of length n_components
    """
    if stdata.cell_type_counts is None:
        raise RuntimeError("SpatialExpressionDataset contains no cell type counts")

    if cell_type_names is None:
        cell_type_names = [f"Cell Type {i}" for i in range(stdata.n_cell_types)]

    fig, ax = plt.subplots()

    for cell_type_idx, cell_type_name in enumerate(cell_type_names):
        output_path = os.path.join(
            output_dir, f"cell_num_one_vs_all_scatterpie__{cell_type_name}.pdf"
        )

        cell_type_counts = np.zeros((stdata.cell_type_counts.shape[0], 2))
        cell_type_counts[:, 0] = stdata.cell_type_counts[:, cell_type_idx]
        for i in range(stdata.n_cell_types):
            if i != cell_type_idx:
                cell_type_counts[:, 1] += stdata.cell_type_counts[:, i]

        common.plot_spatial_pie_charts(
            fig,
            ax,
            stdata.positions_tissue,
            values=cell_type_counts,
            layout=stdata.layout,
            plotting_coordinates=stdata.positions,
            cell_type_names=[cell_type_name, "All Other Cell Types"],
        )
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def plot_loss(deconvolution_results: data.DeconvolutionResult, output_path: str):
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(len(deconvolution_results.losses)), deconvolution_results.losses)
    ax.set_xlabel("Step Number")
    ax.set_ylabel("Loss")
    fig.savefig(output_path)


def rank_genes_groups_plot(
    stdata: data.SpatialExpressionDataset,
    cell_type_labels: Optional[List[str]],
    output_path: str,
    n_genes: int = 15,
    fontsize: int = 8,
    ncols: int = 4,
    sharey: bool = True,
    ax=None,
):
    if stdata.marker_gene_indices is None:
        raise RuntimeError("SpatialExpressionDataset contains no marker genes")

    if stdata.omega_difference is None:
        raise RuntimeError("SpatialExpressionDataset contains no omega difference")

    if cell_type_labels is None:
        group_names = ["Cell Type {}".format(i + 1) for i in range(stdata.n_cell_types)]
    else:
        group_names = cell_type_labels

    n_panels_per_row = ncols
    if n_genes < 1:
        raise NotImplementedError(
            "Specifying a negative number for n_genes has not been implemented for "
            f"this plot. Received n_genes={n_genes}."
        )

    # one panel for each group
    # set up the figure
    n_panels_x = min(n_panels_per_row, len(group_names))
    n_panels_y = np.ceil(len(group_names) / n_panels_x).astype(int)

    fig = plt.figure(
        figsize=(
            n_panels_x * 3,
            n_panels_y * 3,
        )
    )
    gs = gridspec.GridSpec(nrows=n_panels_y, ncols=n_panels_x, wspace=0.22, hspace=0.3)

    ax0 = None
    ymin = np.Inf
    ymax = -np.Inf
    for celltype_idx, marker_gene_set in enumerate(stdata.marker_gene_indices):
        gene_names = stdata.gene_names[marker_gene_set]
        scores = stdata.omega_difference[celltype_idx][marker_gene_set]
        sorted_order = np.argsort(scores)[::-1]

        scores = scores[sorted_order][:n_genes]
        gene_names = gene_names[sorted_order][:n_genes]

        # Setting up axis, calculating y bounds
        if sharey:
            ymin = min(ymin, np.min(scores))
            ymax = max(ymax, np.max(scores))

            if ax0 is None:
                ax = fig.add_subplot(gs[celltype_idx])
                ax0 = ax
            else:
                ax = fig.add_subplot(gs[celltype_idx], sharey=ax0)
        else:
            ymin = np.min(scores)
            ymax = np.max(scores)
            ymax += 0.3 * (ymax - ymin)

            ax = fig.add_subplot(gs[celltype_idx])
            ax.set_ylim(ymin, ymax)

        ax.set_xlim(-0.9, n_genes - 0.1)

        # Making labels
        for ig, gene_name in enumerate(gene_names):
            ax.text(
                ig,
                scores[ig],
                gene_name,
                rotation="vertical",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=fontsize,
            )

        ax.set_title(f"{group_names[celltype_idx]}")
        if celltype_idx >= n_panels_x * (n_panels_y - 1):
            ax.set_xlabel("ranking")

        # print the 'score' label only on the first panel per row.
        if celltype_idx % n_panels_x == 0:
            ax.set_ylabel("score")

    if sharey is True:
        ymax += 0.3 * (ymax - ymin)
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(output_path, pad_inches=0.5)


def plot_deconvolution(
    stdata: data.SpatialExpressionDataset,
    output_dir: str,
    output_format: str = "pdf",
    cell_type_names: Optional[List[str]] = None,
):
    """
    Create a suite of plots for deconvolution results.

    :param stdata: SpatialExpressionDataset to plot
    :param output_dir: Output directory where plots will be saved
    :param output_format: File format of plots
    :param cell_type_names: Cell type names to use in plot, an array of length n_components
    """
    plot_cell_num(
        stdata=stdata,
        output_dir=output_dir,
        output_format=output_format,
        seperate_pdf=False,
        cell_type_names=cell_type_names,
    )

    plot_cell_prob(
        stdata=stdata,
        output_dir=output_dir,
        output_format=output_format,
        seperate_pdf=False,
        cell_type_names=cell_type_names,
    )

    plot_marker_genes(
        stdata=stdata,
        output_file=os.path.join(output_dir, f"marker_genes.{output_format}"),
        cell_type_labels=cell_type_names,
    )

    rank_genes_groups_plot(
        stdata=stdata,
        cell_type_labels=cell_type_names,
        output_path=os.path.join(output_dir, f"rank_genes_groups.{output_format}"),
    )

    plot_cell_num_scatterpie(
        stdata=stdata,
        output_path=os.path.join(output_dir, f"cell_num_scatterpie.{output_format}"),
        cell_type_names=cell_type_names,
    )

    plot_one_vs_all_cell_num_scatterpie(
        stdata=stdata,
        output_dir=output_dir,
        cell_type_names=cell_type_names,
    )
