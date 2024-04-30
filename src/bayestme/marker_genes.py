import os
from enum import Enum
from typing import List, Optional
import logging
import numpy as np
import pandas
from matplotlib import cm as cm, pyplot as plt, colors

from bayestme import data

logger = logging.getLogger(__name__)


class MarkerGeneMethod(Enum):
    TIGHT = "TIGHT"
    BEST_AVAILABLE = "BEST_AVAILABLE"
    FALSE_DISCOVERY_RATE = "FALSE_DISCOVERY_RATE"

    def __str__(self):
        return self.value


def select_marker_genes(
    deconvolution_result: data.DeconvolutionResult,
    n_marker: int = 5,
    alpha: float = 0.05,
    method: MarkerGeneMethod = MarkerGeneMethod.TIGHT,
) -> List[np.ndarray]:
    """
    Returns a list of length <N components>, where
    the values are indices of the marker genes in the gene name array for
    that component.

    :param deconvolution_result: DeconvolutionResult object
    :param n_marker: Number of markers per cell type to select
    :param alpha: Marker gene threshold parameter, defaults to 0.05
    :param method: Enum representing which marker gene selection method to use.
    :return: List of arrays of gene indices
    """
    marker_gene_sets = []
    for k in range(deconvolution_result.n_components):
        # fdr control
        if method is MarkerGeneMethod.TIGHT:
            marker_control = deconvolution_result.omega[k] > 1 - alpha
            marker_idx_control = np.argwhere(marker_control).flatten()
            if marker_idx_control.sum() == 0:
                raise RuntimeError(
                    "0 genes satisfy omega > 1 - {}. Only {} genes satify the condition.".format(
                        alpha, marker_idx_control.sum()
                    )
                )
        elif method is MarkerGeneMethod.FALSE_DISCOVERY_RATE:
            sorted_index = np.argsort(1 - deconvolution_result.omega[k])
            fdr = np.cumsum(1 - deconvolution_result.omega[k][sorted_index]) / (
                np.arange(sorted_index.shape[0]) + 1
            )
            marker_control = np.argwhere(fdr <= alpha).flatten()
            marker_idx_control = sorted_index[marker_control]
        elif method is MarkerGeneMethod.BEST_AVAILABLE:
            marker_idx_control = np.arange(deconvolution_result.omega[k].shape[0])
        else:
            raise ValueError(method)
        # sort adjointly by omega_kg (primary) and expression level (secondary)
        top_marker = np.lexsort(
            (
                deconvolution_result.relative_expression[k][marker_idx_control],
                deconvolution_result.omega[k][marker_idx_control],
            )
        )[::-1]

        if n_marker > len(top_marker):
            logger.warning(
                f"For cell type ({k}) fewer then ({n_marker}) genes "
                f"met the marker gene criteria, will only use "
                f"{len(top_marker)} marker genes for this cell type."
            )

        marker_gene_sets.append(marker_idx_control[top_marker[:n_marker]])

    return marker_gene_sets


def add_marker_gene_results_to_dataset(
    stdata: data.SpatialExpressionDataset, marker_genes: List[np.ndarray]
):
    """
    Modify stdata in place to to annotate it with marker gene selection results.

    :param stdata: data.SpatialExpressionDataset to modify
    :param marker_genes: Selected marker genes to add to dataset
    """
    marker_gene_boolean = np.zeros((stdata.n_gene, stdata.n_cell_types)).astype(int)
    marker_gene_boolean[:, :] = -1

    for i in range(stdata.n_cell_types):
        marker_gene_boolean[:, i][marker_genes[i]] = np.arange(marker_genes[i].size)

    stdata.adata.varm[data.MARKER_GENE_ATTR] = marker_gene_boolean


def plot_marker_genes(
    stdata: data.SpatialExpressionDataset,
    output_file: str,
    cell_type_labels: Optional[List[str]] = None,
    colormap: cm.ScalarMappable = cm.BuPu,
):
    if stdata.marker_gene_indices is None:
        raise RuntimeError("SpatialExpressionDataset contains no marker genes")

    if stdata.omega_difference is None:
        raise RuntimeError("SpatialExpressionDataset contains no omega difference")

    all_gene_indices = np.concatenate(stdata.marker_gene_indices)
    all_gene_names = stdata.gene_names[all_gene_indices]
    n_marker = len(all_gene_indices)

    if cell_type_labels is None:
        cell_type_labels = [
            "Cell Type {}".format(i + 1) for i in range(stdata.n_cell_types)
        ]

    fig, (ax_genes, ax_legend) = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [n_marker, 2]}
    )
    inches_per_column = 0.75

    fig.set_figwidth(max(fig.get_size_inches()[0], (n_marker * inches_per_column)))

    offset = 0
    divider_lines = []
    for k, marker_gene_set in enumerate(stdata.marker_gene_indices):
        divider_lines.append(offset)
        vmin = min(-1e-4, stdata.omega_difference[k][all_gene_indices].min())
        vmax = max(1e-4, stdata.omega_difference[k][all_gene_indices].max())
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        ax_genes.scatter(
            np.arange(n_marker),
            np.ones(n_marker) * (k + 1),
            c=stdata.omega_difference[k][all_gene_indices],
            s=norm(abs(stdata.omega_difference[k][all_gene_indices]))
            * inches_per_column
            * fig.dpi
            * 3,
            cmap=colormap,
            norm=norm,
        )
        offset = offset + len(marker_gene_set)
    divider_lines.append(offset)

    ax_genes.set_xticks(np.arange(n_marker))
    ax_genes.set_xticklabels(
        all_gene_names, fontweight="bold", rotation=45, ha="right", va="top"
    )
    ax_genes.set_yticks([x + 1 for x in range(stdata.n_cell_types)])
    ax_genes.set_yticklabels(cell_type_labels, rotation=0, fontweight="bold")
    ax_genes.invert_yaxis()
    ax_genes.margins(x=0.02, y=0.1)
    for x in divider_lines:
        ax_genes.axvline(x - 0.5, ls="--", c="k", alpha=0.5)

    legend_values = np.array([0.25, 0.5, 0.75, 1])
    norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    ax_legend.scatter(
        np.ones(len(legend_values)),
        np.arange(len(legend_values)),
        cmap=colormap,
        c=legend_values,
        s=legend_values * inches_per_column * fig.dpi * 3,
        norm=norm,
    )

    ax_legend.set_yticks(np.arange(len(legend_values)))
    ax_legend.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontweight="bold")
    ax_legend.yaxis.tick_right()
    ax_legend.set_xticks([])
    ax_legend.spines["top"].set_visible(False)
    ax_legend.spines["right"].set_visible(False)
    ax_legend.spines["bottom"].set_visible(False)
    ax_legend.spines["left"].set_visible(False)
    ax_legend.set_ylabel(
        "Loading", rotation=270, labelpad=50, fontweight="bold", fontsize="25"
    )
    ax_legend.yaxis.set_label_position("right")
    ax_legend.margins(y=0.2)

    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")

    plt.close(fig)


def create_top_gene_lists(
    stdata: data.SpatialExpressionDataset,
    deconvolution_result: data.DeconvolutionResult,
    output_path: str,
    n_marker_genes: int = 5,
    alpha: float = 0.05,
    marker_gene_method: MarkerGeneMethod = MarkerGeneMethod.TIGHT,
    cell_type_names=None,
):
    marker_genes = select_marker_genes(
        deconvolution_result=deconvolution_result,
        n_marker=n_marker_genes,
        alpha=alpha,
        method=marker_gene_method,
    )

    results = []
    for k, marker_gene_set in enumerate(marker_genes):
        result = pandas.DataFrame()
        result["gene_name"] = stdata.gene_names[marker_gene_set]

        result["rank_in_cell_type"] = np.arange(0, len(marker_gene_set))

        if cell_type_names is None:
            result["cell_type"] = np.repeat(np.array([k + 1]), n_marker_genes)
        else:
            result["cell_type"] = np.repeat(
                np.array(cell_type_names[k]), n_marker_genes
            )
        results.append(result)

    pandas.concat(results).to_csv(output_path, header=True, index=False)


def create_marker_gene_ranking_csvs(
    stdata: data.SpatialExpressionDataset,
    deconvolution_result: data.DeconvolutionResult,
    output_dir: str,
):
    relative_expression_df = pandas.DataFrame(
        index=stdata.gene_names,
        columns=range(deconvolution_result.n_components),
    )

    omega_df = pandas.DataFrame(
        index=stdata.gene_names,
        columns=range(deconvolution_result.n_components),
    )

    for k in range(deconvolution_result.n_components):
        relative_expression_df[k] = deconvolution_result.relative_expression[k]
        omega_df[k] = deconvolution_result.omega[k]

    relative_expression_df.to_csv(
        os.path.join(output_dir, "relative_expression.csv"), index=False
    )
    omega_df.to_csv(os.path.join(output_dir, "omega.csv"), index=False)
