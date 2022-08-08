import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.axes
import matplotlib.figure

from typing import Tuple

from matplotlib.colors import Normalize
from matplotlib.patches import RegularPolygon, Wedge, Patch

from bayestme import data


def get_x_y_arrays_for_layout(coords: np.ndarray, layout: data.Layout) -> Tuple[np.array, np.array]:
    if layout is data.Layout.HEX:
        hcoord = coords[:, 0]
        vcoord = (2. * np.sin(np.radians(60)) * (coords[:, 1]) / 3.)
    elif layout is data.Layout.SQUARE:
        hcoord = coords[:, 0]
        vcoord = coords[:, 1]
    else:
        raise NotImplementedError(layout)
    return hcoord, vcoord


def plot_colored_spatial_polygon(
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        coords: np.ndarray,
        values: np.ndarray,
        layout: data.Layout,
        colormap: cm.ScalarMappable = cm.BuPu,
        norm=None,
        plotting_coordinates=None,
        normalize=True):
    """
    Basic plot of spatial gene expression


    :param fig: matplotlib figure artist object to which plot will be written
    :param ax: matplotlib axes artist object to which plot will be written
    :param coords: np.ndarray of int, shape of (N, 2)
    :param values: np.ndarray of int, shape of (N,)
    :param layout: Layout enum
    :param colormap: Colormap for converting values to colors, defaults to BuPu
    :param norm: Function for normalizing scalar values, defaults to Normalizer over values domain
    :param plotting_coordinates: Expand the plotting window to include these coordinates,
                                 default is to just plot over coords.
    :param normalize: Whether to normalize values before coloring them or not. Set false for boolean data.
    :return: matplotlib Figure object
    """
    if norm is None:
        norm = Normalize(vmin=np.min(values), vmax=np.max(values))

    ax.set_aspect('equal')
    hcoord, vcoord = get_x_y_arrays_for_layout(coords, layout)
    if plotting_coordinates is None:
        support_hcoord, support_vcoord = (hcoord, vcoord)
    else:
        support_hcoord, support_vcoord = get_x_y_arrays_for_layout(plotting_coordinates, layout)

    if layout is data.Layout.HEX:
        num_vertices = 6
        packing_radius = 2.0 / 3.0
        orientation = np.radians(30)
    elif layout is data.Layout.SQUARE:
        num_vertices = 4
        packing_radius = math.sqrt(2) / 2.0
        orientation = np.radians(45)
    else:
        raise NotImplementedError(layout)

    # Add colored polygons
    for x, y, v in zip(hcoord, vcoord, values):
        polygon = RegularPolygon(
            (x, y),
            numVertices=num_vertices,
            radius=packing_radius,
            orientation=orientation,
            facecolor=colormap(norm(v)) if normalize else colormap(v),
            alpha=1,
            edgecolor='k',
            linewidth=0.3)
        ax.add_patch(polygon)

    # By scatter-plotting an invisible point on to all of our patches
    # we ensure the plotting domain is
    # adjusted such that all patches are visible.
    ax.scatter(support_hcoord, support_vcoord, alpha=0)

    cb = fig.colorbar(cm.ScalarMappable(norm=norm if normalize else None, cmap=colormap), ax=ax)

    return ax, cb, norm, hcoord, vcoord


def get_wedge_dimensions_from_value_array(value_array: np.array):
    """
    Get a series of N (start, stop) pairs in degrees of the pie chart defined
    by the data in the length N value_array
    :param value_array: np.array of length N
    :return: list of 2-tuples of length N.
    """
    if value_array.sum() == 0:
        value_array = value_array + 1

    theta2_values = np.cumsum((value_array / value_array.sum()) * 360.0)
    theta1_values = np.concatenate([[0], theta2_values[:-1]])

    return list(zip(theta1_values, theta2_values))


def plot_spatial_pie_charts(
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        coords: np.ndarray,
        values: np.ndarray,
        layout: data.Layout,
        colormap: cm.ScalarMappable = cm.Set1,
        plotting_coordinates=None,
        cell_type_names=None):
    """
    Plot pie charts to show relative proportions of multiple scalar values at each spot

    :param fig: matplotlib figure artist object to which plot will be written
    :param ax: matplotlib axes artist object to which plot will be written
    :param coords: np.ndarray of int, shape of (N, 2)
    :param values: np.ndarray of int, shape of (N, M)
    :param layout: Layout enum
    :param colormap: Colormap for the pie chart wedges, defaults to Set1
    :param plotting_coordinates: Expand the plotting window to include these coordinates,
                                 default is to just plot over coords.
    :param cell_type_names: A array of length n_components, which provides cell type names.
    :return: matplotlib Figure object
    """

    ax.set_aspect('equal')

    hcoord, vcoord = get_x_y_arrays_for_layout(coords, layout)
    if plotting_coordinates is None:
        support_hcoord, support_vcoord = (hcoord, vcoord)
    else:
        support_hcoord, support_vcoord = get_x_y_arrays_for_layout(plotting_coordinates, layout)

    if layout is data.Layout.HEX:
        packing_radius = .5
    elif layout is data.Layout.SQUARE:
        packing_radius = .5
    else:
        raise NotImplementedError(layout)

    # Add colored polygons
    for x, y, vs in zip(hcoord, vcoord, values):
        for idx, (theta1, theta2) in enumerate(get_wedge_dimensions_from_value_array(vs)):
            wedge = Wedge(
                center=(x, y),
                r=packing_radius,
                theta1=theta1,
                theta2=theta2,
                facecolor=colormap(idx),
                alpha=1)
            ax.add_patch(wedge)

    # By scatter-plotting an invisible point on to all of our patches
    # we ensure the plotting domain is
    # adjusted such that all patches are visible.
    ax.scatter(support_hcoord, support_vcoord, alpha=0)

    # create a patch (proxy artist) for every color
    patches = []
    for i in range(values.shape[1]):
        if cell_type_names is not None:
            label = cell_type_names[i]
        else:
            label = f'Cell Type {i + 1}'
        patches.append(Patch(color=colormap(i), label=label))

    # put those patched as legend-handles into the legend
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ax.set_axis_off()
    return ax, hcoord, vcoord


def plot_gene_in_tissue_counts(stdata: data.SpatialExpressionDataset,
                               gene: str):
    gene_idx = np.argmax(stdata.gene_names == gene)
    counts = stdata.raw_counts[:, gene_idx]
    counts = counts[stdata.tissue_mask]
    positions = stdata.positions[stdata.tissue_mask, :]

    fig, ax = plt.subplots(1)

    ax, cb, norm, hcoord, vcoord = plot_colored_spatial_polygon(
        fig=fig,
        ax=ax,
        coords=positions,
        values=counts,
        layout=stdata.layout,
        plotting_coordinates=stdata.positions)

    return fig, ax, cb, norm, hcoord, vcoord


def plot_gene_raw_counts(stdata: data.SpatialExpressionDataset,
                         gene: str,
                         output_file: str):
    gene_idx = np.argmax(stdata.gene_names == gene)

    fig, ax = plt.subplots(1)

    plot_colored_spatial_polygon(
        fig=fig,
        ax=ax,
        coords=stdata.positions,
        values=stdata.raw_counts[:, gene_idx],
        layout=stdata.layout)

    fig.savefig(output_file)
    plt.close(fig)
