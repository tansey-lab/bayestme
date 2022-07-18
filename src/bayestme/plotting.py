import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import matplotlib.axes
import matplotlib.figure
from matplotlib.colors import Normalize

from matplotlib.patches import RegularPolygon, Wedge, Patch

from bayestme import data
import math

from typing import Tuple


def st_plot(data,
            pos,
            cmap='BuPu',
            v_min=None,
            v_max=None,
            norm=None,
            layout='s',
            unit_dist=10,
            x_y_swap=False,
            invert=[0, 0],
            name='st_plot',
            colorbar=True,
            subtitles=None,
            save='.',
            plot_format='pdf'):
    if x_y_swap:
        pos = pos[::-1]
    n_plots = data.shape[0]
    if not isinstance(v_min, (list, np.ndarray)):
        v_min = [v_min for i in range(n_plots)]
    if not isinstance(v_max, (list, np.ndarray)):
        v_max = [v_max for i in range(n_plots)]
    subplots_adj = 1 / n_plots
    x_axis_distance = pos[0].max() - pos[0].min() + 2
    y_axis_distance = pos[1].max() - pos[1].min() + 2
    dpi = plt.rcParams["figure.dpi"]
    text_width = plt.rcParams['font.size'] * 2
    if layout == 'H' and invert[0] + invert[1] == 1:
        layout = 'h'
        text_ratio = text_width / (x_axis_distance * unit_dist + text_width)
        st_ratio = 1 - text_ratio - 0.08
    else:
        text_ratio = text_width / (x_axis_distance * np.sqrt(3) * unit_dist + text_width)
        st_ratio = 1 - text_ratio - 0.1

    if subtitles:
        h = 0.78

    else:
        h = 1
    h_cb_l = h * 0.03
    if layout == 's':
        scatter_size = unit_dist ** 2
        fig_width = (x_axis_distance * unit_dist / dpi) / st_ratio
        fig_height = y_axis_distance * unit_dist / dpi / h
    elif layout == 'H':
        scatter_size = (2 * unit_dist) ** 2
        fig_width = x_axis_distance * np.sqrt(3) * unit_dist / dpi / st_ratio
        fig_height = y_axis_distance * unit_dist / dpi / h
    elif layout == 'h':
        scatter_size = (2 * unit_dist) ** 2
        fig_width = x_axis_distance * unit_dist / dpi / st_ratio
        fig_height = y_axis_distance * np.sqrt(3) * unit_dist / dpi / h
    fig = plt.figure(figsize=(fig_width * n_plots, fig_height))
    for i in range(n_plots):
        stbox = [0 + i * subplots_adj, 0, st_ratio * subplots_adj, h]
        cbbox = [(st_ratio + 0.01 + i) * subplots_adj, h_cb_l, 0.04 * subplots_adj, h - h_cb_l * 2]
        stframe = plt.axes(stbox)
        img = stframe.scatter(pos[0], pos[1], c=data[i][0], cmap=cmap, s=scatter_size, vmin=v_min[i], vmax=v_max[i],
                              norm=norm, marker=layout, linewidths=0)
        if data[i].shape[0] > 1:
            stframe.scatter(pos[0], pos[1], c=data[i][1], cmap=cmap, s=scatter_size, alpha=0.2, vmin=v_min[i],
                            vmax=v_max[i], norm=norm,
                            marker=layout, linewidths=0)
        if colorbar:
            cbframe = plt.axes(cbbox)
            plt.colorbar(img, cax=cbframe)
        stframe.set_xlim(pos[0].min() - 1, pos[0].max() + 1)
        stframe.set_ylim(pos[1].min() - 1, pos[1].max() + 1)
        stframe.axis('off')
        if subtitles:
            stframe.set_title(subtitles[i], fontweight='bold')
        if invert[0]:
            stframe.invert_xaxis()
        if invert[1]:
            stframe.invert_yaxis()

    print(f'Plot saved as {save}/{name}.{plot_format}')
    plt.tight_layout()
    plt.savefig(os.path.join(save, f'{name}.{plot_format}'), bbox_inches='tight')
    plt.close()


def plot_spots(ax, data, pos, rgb=np.array([1, 0, 0]), cmap=None, discrete_cmap=None, s=15, v_min=None, v_max=None,
               invert_x=True, invert_y=True, norm=None):
    if cmap is None and discrete_cmap is None:
        n_nodes = data.shape[0]
        plot_color = np.zeros((n_nodes, 4))
        plot_color[:, :3] = rgb[None]
        plot_color[:, 3] = data / data.max()
        ax.scatter(pos[0], pos[1], color=np.array([0, 0, 0, 0.02]), edgecolors=None, linewidths=None, s=s)
        img = ax.scatter(pos[0], pos[1], color=plot_color, edgecolors=None, linewidths=0, s=s)
    elif discrete_cmap is not None:
        for i, value in enumerate(np.unique(data)):
            idx = np.argwhere(data == value).flatten()
            img = ax.scatter(pos[0, idx], pos[1, idx], color=discrete_cmap[value], edgecolors=None, linewidths=0, s=s)
    else:
        img = ax.scatter(pos[0], pos[1], c=data, cmap=cmap, s=s, vmin=v_min, vmax=v_max, norm=norm)
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    return img


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
        plotting_coordinates=None):
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
    :return: matplotlib Figure object
    """
    if norm is None:
        norm = Normalize(vmin=np.min(values), vmax=np.max(values))

    ax.set_aspect('equal')
    hcoord, vcoord = get_x_y_arrays_for_layout(coords, layout)
    if plotting_coordinates is None:
        support_hcoord, support_vcoord = (hcoord, vcoord)
    else:
        support_hcoord, support_vcoord = get_x_y_arrays_for_layout(coords, layout)

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
            facecolor=colormap(norm(v)),
            alpha=1,
            edgecolor='k')
        ax.add_patch(polygon)

    # By scatter-plotting an invisible point on to all of our patches
    # we ensure the plotting domain is
    # adjusted such that all patches are visible.
    ax.scatter(support_hcoord, support_vcoord, alpha=0)

    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)

    return ax, cb, norm, hcoord, vcoord


def get_wedge_dimensions_from_value_array(value_array: np.array):
    """
    Get a series of N (start, stop) pairs in degrees of the pie chart defined
    by the data in the length N value_array
    :param value_array: np.array of length N
    :return: list of 2-tuples of length N.
    """
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
        plotting_coordinates=None):
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
    :return: matplotlib Figure object
    """

    ax.set_aspect('equal')

    hcoord, vcoord = get_x_y_arrays_for_layout(coords, layout)
    if plotting_coordinates is None:
        support_hcoord, support_vcoord = (hcoord, vcoord)
    else:
        support_hcoord, support_vcoord = get_x_y_arrays_for_layout(coords, layout)

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
        patches.append(Patch(color=colormap(i), label=f'Cell Type {i + 1}'))

    # put those patched as legend-handles into the legend
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ax.set_axis_off()
    return ax, hcoord, vcoord


def plot_gene_in_tissue_counts(stdata: data.SpatialExpressionDataset,
                               gene: str,
                               output_file: str):
    gene_idx = np.argmax(stdata.gene_names == gene)
    counts = stdata.raw_counts[:, gene_idx]
    counts = counts[stdata.tissue_mask]
    positions = stdata.positions.T[stdata.tissue_mask, :]

    fig, ax = plt.subplots(1)

    plot_colored_spatial_polygon(
        fig=fig,
        ax=ax,
        coords=positions,
        values=counts,
        layout=stdata.layout,
        plotting_coordinates=stdata.positions.T)

    fig.savefig(output_file)
    plt.close(fig)


def plot_gene_raw_counts(stdata: data.SpatialExpressionDataset,
                         gene: str,
                         output_file: str):
    gene_idx = np.argmax(stdata.gene_names == gene)

    fig, ax = plt.subplots(1)

    plot_colored_spatial_polygon(
        fig=fig,
        ax=ax,
        coords=stdata.positions.T,
        values=stdata.raw_counts[:, gene_idx],
        layout=stdata.layout)

    fig.savefig(output_file)
    plt.close(fig)
