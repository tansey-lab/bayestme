import os
from typing import Optional, List

from matplotlib import cm as cm, pyplot as plt

from bayestme import data
from bayestme.marker_genes import plot_marker_genes
from bayestme.plot import common


def plot_cell_num(
    stdata: data.SpatialExpressionDataset,
    output_dir: str,
    output_format: str = "pdf",
    cmap=cm.jet,
    seperate_pdf: bool = False,
    cell_type_names: Optional[List[str]] = None,
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
        fig, axes = plt.subplots(
            ncols=stdata.n_cell_types, subplot_kw=dict(adjustable="box", aspect="equal")
        )
        fig.set_figwidth(fig.get_size_inches()[0] * stdata.n_cell_types)

        for i, ax in enumerate(axes):
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
        fig.savefig(os.path.join(output_dir, f"cell_type_counts.{output_format}"))
        plt.close(fig)


def plot_cell_prob(
    stdata: data.SpatialExpressionDataset,
    output_dir: str,
    output_format: str = "pdf",
    cmap=cm.jet,
    seperate_pdf: bool = False,
    cell_type_names: Optional[List[str]] = None,
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
        fig, axes = plt.subplots(
            ncols=stdata.n_cell_types, subplot_kw=dict(adjustable="box", aspect="equal")
        )

        fig.set_figwidth(fig.get_size_inches()[0] * stdata.n_cell_types)

        for i, ax in enumerate(axes):
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

    plot_cell_num_scatterpie(
        stdata=stdata,
        output_path=os.path.join(output_dir, f"cell_num_scatterpie.{output_format}"),
        cell_type_names=cell_type_names,
    )
