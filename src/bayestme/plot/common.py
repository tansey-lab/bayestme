import math
from typing import Tuple

import matplotlib.axes
import matplotlib.cm as cm
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib.patches import RegularPolygon, Wedge, Patch, Polygon
from scipy.spatial import Voronoi
import networkx as nx

import shapely
from matplotlib import patches

import bayestme.common
from bayestme.common import Layout
from bayestme import data

from geopandas import GeoDataFrame

# Extended version of cm.Set1 categorical colormap. See references:
#
#     ¹) Glasbey, C., van der Heijden, G., Toh, V. F. K. and Gray, A. (2007),
#        Colour Displays for Categorical Images.
#        Color Research and Application, 32: 304-309
#
#     ²) Luo, M. R., Cui, G. and Li, C. (2006),
#        Uniform Colour Spaces Based on CIECAM02 Colour Appearance Model.
#        Color Research and Application, 31: 320–330
GLASBEY_30_COLORS = [
    "#e51415",
    "#347fba",
    "#4bb049",
    "#9a4da4",
    "#ff8000",
    "#ffff30",
    "#a85523",
    "#f782c0",
    "#9b9b9b",
    "#004100",
    "#00ffff",
    "#000090",
    "#5f0025",
    "#fff5ff",
    "#4d4d50",
    "#b59bff",
    "#6700f9",
    "#4c006f",
    "#00816c",
    "#72ff9c",
    "#6a6c00",
    "#e2c88c",
    "#63c3fe",
    "#da0084",
    "#593200",
    "#40bcab",
    "#b39c00",
    "#f200ff",
    "#930000",
    "#004d74",
]
Glasbey30 = ListedColormap(colors=GLASBEY_30_COLORS, name="Glasbey30", N=30)


def get_x_y_arrays_for_layout(
    coords: np.ndarray, layout: bayestme.common.Layout
) -> Tuple[np.array, np.array]:
    if layout is Layout.HEX:
        hcoord = coords[:, 0]
        vcoord = coords[:, 1]
    elif layout is Layout.SQUARE:
        hcoord = coords[:, 0]
        vcoord = coords[:, 1]
    elif layout is Layout.IRREGULAR:
        hcoord = coords[:, 0]
        vcoord = coords[:, 1]
    else:
        raise NotImplementedError(layout)
    return hcoord, vcoord


def mirror_points(points, xlim, ylim):
    """Mirror points for reflective boundary conditions"""
    mirrored_points = np.vstack(
        [
            points,
            np.vstack([points[:, 0], ylim - points[:, 1]]).T,  # Top boundary
            np.vstack([points[:, 0], -points[:, 1]]).T,  # Bottom boundary
            np.vstack([xlim - points[:, 0], points[:, 1]]).T,  # Right boundary
            np.vstack([-points[:, 0], points[:, 1]]).T,
        ]
    )  # Left boundary
    return mirrored_points


def get_voronoi_poly_and_vertices(vor):
    for region_index in vor.point_region:
        region = vor.regions[region_index]
        if not -1 in region:  # Bounded regions
            polygon = shapely.Polygon([vor.vertices[i] for i in region])
            yield polygon, [vor.vertices[i] for i in region]


def get_mirrored_voronoi_poly_and_vertices(points):
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    points_as_tuples_mirrored = mirror_points(points, x_max - x_min, y_max - y_min)
    for poly, vertices in get_voronoi_poly_and_vertices(
        Voronoi(points_as_tuples_mirrored)
    ):
        yield poly, vertices


def add_voronoi_cell_bodies(ax, points, values, colormap, norm=None):
    gdf = GeoDataFrame()
    gdf["centroid"] = [shapely.Point(*a) for a in points]
    gdf["value"] = values
    gdf = gdf.set_geometry("centroid")

    for poly, vertices in get_mirrored_voronoi_poly_and_vertices(points):
        selector = gdf.within(poly)

        if selector.sum() == 0:
            continue
        elif selector.sum() == 1:
            value = gdf[selector]["value"]
            patch = patches.Polygon(
                np.array(vertices),
                facecolor=colormap(norm(value)) if norm else colormap(value),
                alpha=1,
                edgecolor="black",
                linewidth=0.3,
            )
            ax.add_patch(patch)
        else:
            raise RuntimeError("Multiple nuclei in one voronoi cell")


def add_voronoi_bounded_scatterpies(ax, points, values, colormap):
    gdf = GeoDataFrame()
    gdf["centroid"] = [shapely.Point(*a) for a in points]

    n_cell_types = values.shape[1]

    for i in range(n_cell_types):
        gdf[f"cell_type_{i}"] = values[:, i]

    gdf = gdf.set_geometry("centroid")

    for poly, vertices in get_mirrored_voronoi_poly_and_vertices(points):
        selector = gdf.within(poly)

        if selector.sum() == 0:
            continue
        elif selector.sum() == 1:
            values = []
            for i in range(n_cell_types):
                values.append(gdf[selector][f"cell_type_{i}"].to_numpy().item())
            centroid = poly.centroid
            radius = radius_of_largest_circle_centered_inside_polygon(poly)

            for idx, (theta1, theta2) in enumerate(
                get_wedge_dimensions_from_value_array(np.array(values))
            ):
                wedge = Wedge(
                    center=(centroid.x, centroid.y),
                    r=radius,
                    theta1=theta1,
                    theta2=theta2,
                    facecolor=colormap(idx),
                    alpha=1,
                )
                ax.add_patch(wedge)
        else:
            raise RuntimeError("Multiple nuclei in one voronoi cell")


def plot_colored_spatial_polygon(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    coords: np.ndarray,
    values: np.ndarray,
    layout: bayestme.common.Layout,
    colormap: cm.ScalarMappable = cm.BuPu,
    norm=None,
    plotting_coordinates=None,
    normalize=True,
    border_mask=None,
):
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
    if border_mask is None:
        border_mask = np.ones_like(values, dtype=bool)

    ax.set_aspect("equal")
    hcoord, vcoord = get_x_y_arrays_for_layout(coords, layout)
    if plotting_coordinates is None:
        support_hcoord, support_vcoord = (hcoord, vcoord)
    else:
        support_hcoord, support_vcoord = get_x_y_arrays_for_layout(
            plotting_coordinates, layout
        )

    if layout in [Layout.HEX, Layout.SQUARE]:
        if layout is Layout.HEX:
            num_vertices = 6
            packing_radius = 2.0 / 3.0
            orientation = np.radians(30)
        elif layout is Layout.SQUARE:
            num_vertices = 4
            packing_radius = math.sqrt(2) / 2.0
            orientation = np.radians(45)
        else:
            raise NotImplementedError(layout)

        # Add colored polygons
        for x, y, v, has_border in zip(hcoord, vcoord, values, border_mask):
            polygon = RegularPolygon(
                (x, y),
                numVertices=num_vertices,
                radius=packing_radius,
                orientation=orientation,
                facecolor=colormap(norm(v)) if normalize else colormap(v),
                alpha=1,
                edgecolor="black" if has_border else "white",
                linewidth=0.1,
            )
            ax.add_patch(polygon)
    elif layout is Layout.IRREGULAR:
        add_voronoi_cell_bodies(ax, coords, values, colormap, norm=norm)

    # By scatter-plotting an invisible point on to all of our patches
    # we ensure the plotting domain is
    # adjusted such that all patches are visible.
    ax.scatter(support_hcoord, support_vcoord, alpha=0)

    cb = fig.colorbar(
        cm.ScalarMappable(norm=norm if normalize else None, cmap=colormap), ax=ax
    )

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


def radius_of_largest_circle_centered_inside_polygon(polygon: shapely.Polygon):
    # Calculate the centroid of the polygon
    centroid = polygon.centroid

    # Initialize the minimum distance to a large number
    min_distance = float("inf")

    # Iterate through each edge of the polygon
    for i in range(len(polygon.exterior.coords) - 1):
        p1 = shapely.Point(polygon.exterior.coords[i])
        p2 = shapely.Point(polygon.exterior.coords[i + 1])

        # Create a line from two consecutive points (an edge)
        line = shapely.LineString([p1, p2])

        # Calculate the perpendicular distance from the centroid to the line
        distance = line.distance(centroid)

        # Update the minimum distance
        min_distance = min(min_distance, distance)

    # The radius of the largest inscribed circle is the minimum distance
    return min_distance


def plot_spatial_pie_charts(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    coords: np.ndarray,
    values: np.ndarray,
    layout: bayestme.common.Layout,
    colormap: cm.ScalarMappable = Glasbey30,
    plotting_coordinates=None,
    cell_type_names=None,
):
    """
    Plot pie charts to show relative proportions of multiple scalar values at each spot

    :param fig: matplotlib figure artist object to which plot will be written
    :param ax: matplotlib axes artist object to which plot will be written
    :param coords: np.ndarray of int, shape of (N, 2)
    :param values: np.ndarray of int, shape of (N, M)
    :param layout: Layout enum
    :param colormap: Colormap for the pie chart wedges, defaults to Glasbey30
    :param plotting_coordinates: Expand the plotting window to include these coordinates,
                                 default is to just plot over coords.
    :param cell_type_names: A array of length n_components, which provides cell type names.
    :return: matplotlib Figure object
    """

    ax.set_aspect("equal")

    hcoord, vcoord = get_x_y_arrays_for_layout(coords, layout)
    if plotting_coordinates is None:
        support_hcoord, support_vcoord = (hcoord, vcoord)
    else:
        support_hcoord, support_vcoord = get_x_y_arrays_for_layout(
            plotting_coordinates, layout
        )

    if layout in [Layout.HEX, Layout.SQUARE]:
        packing_radius = 0.5

        # Add colored polygons
        for x, y, vs in zip(hcoord, vcoord, values):
            for idx, (theta1, theta2) in enumerate(
                get_wedge_dimensions_from_value_array(vs)
            ):
                wedge = Wedge(
                    center=(x, y),
                    r=packing_radius,
                    theta1=theta1,
                    theta2=theta2,
                    facecolor=colormap(idx),
                    alpha=1,
                )
                ax.add_patch(wedge)
    elif layout is Layout.IRREGULAR:
        add_voronoi_bounded_scatterpies(ax, coords, values, colormap)
    else:
        raise NotImplementedError(layout)

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
            label = f"Cell Type {i + 1}"
        patches.append(Patch(color=colormap(i), label=label))

    # put those patched as legend-handles into the legend
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    ax.set_axis_off()
    return ax, hcoord, vcoord


def plot_gene_in_tissue_counts(
    stdata: data.SpatialExpressionDataset, gene: str, output_file: str
):
    gene_idx = np.argmax(stdata.gene_names == gene)
    counts = stdata.raw_counts[:, gene_idx]
    counts = counts[stdata.tissue_mask]
    positions = stdata.positions[stdata.tissue_mask, :]

    fig, ax = plt.subplots(1)

    plot_colored_spatial_polygon(
        fig=fig,
        ax=ax,
        coords=positions,
        values=counts,
        layout=stdata.layout,
        plotting_coordinates=stdata.positions,
    )

    fig.savefig(output_file)
    plt.close(fig)


def plot_gene_raw_counts(
    stdata: data.SpatialExpressionDataset, gene: str, output_file: str
):
    gene_idx = np.argmax(stdata.gene_names == gene)

    fig, ax = plt.subplots(1)

    plot_colored_spatial_polygon(
        fig=fig,
        ax=ax,
        coords=stdata.positions,
        values=stdata.raw_counts[:, gene_idx],
        layout=stdata.layout,
        border_mask=stdata.tissue_mask,
    )

    fig.savefig(output_file)
    plt.close(fig)


def plot_spot_connectivity_graph(node_coordinates, edges, output_fn):
    # Create a graph instance
    fig, ax = plt.subplots(1, figsize=(10, 10))
    G = nx.Graph()

    # Add nodes and their positions to the graph
    for i, (x, y) in enumerate(node_coordinates):
        G.add_node(i, pos=(x, y))

    # Add edges to the graph
    for edge in edges:
        G.add_edge(*edge)

    # Get node positions for plotting
    pos = nx.get_node_attributes(G, "pos")

    # Draw the graph
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color="skyblue",
        node_size=5,
        edge_color="gray",
        ax=ax,
    )

    ax.set_title("Spot Connectivity Graph")

    fig.savefig(output_fn)
