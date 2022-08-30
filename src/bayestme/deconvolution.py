import logging
import numpy as np
import pandas
import os.path

from enum import Enum
from typing import Optional, List
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
from bayestme import model_bkg, data, plotting

logger = logging.getLogger(__name__)


def load_expression_truth(
        stdata: data.SpatialExpressionDataset,
        seurat_output: str):
    """
    Load outputs from seurat fine mapping to be used in deconvolution.

    :param stdata: SpatialExpressionDataset object
    :param seurat_output: CSV output from seurat fine mapping workflow
    :return: Tuple of n_components x n_genes size array, representing relative
    expression of each gene in each cell type
    """
    df = pandas.read_csv(seurat_output, index_col=0)

    phi_k_truth = df.loc[stdata.gene_names].to_numpy()

    # re-normalize so expression values sum to 1 within each component for
    # this subset of genes
    phi_k_truth_normalized = phi_k_truth / phi_k_truth.sum(axis=0)

    return phi_k_truth_normalized.T


def deconvolve(
        reads,
        edges,
        n_gene=None,
        n_components=None,
        lam2=None,
        n_samples=100,
        n_burnin=1000,
        n_thin=10,
        bkg=False,
        lda=False,
        expression_truth=None,
        rng: Optional[np.random.Generator] = None
        ) -> data.DeconvolutionResult:
    """
    Run deconvolution

    :param reads: Read count matrix
    :param edges: Spot adjacency matrix
    :param n_gene: Number of gene markers
    :param n_components: Number of components or cell types
    :param lam2: Lambda smoothing parameter
    :param n_samples: Number of total samples from the posterior distribution
    :param n_burnin: Number of burn in samples before samples are saved
    :param n_thin: Proportion of samples to save
    :param bkg:
    :param lda: If true use LDA initialization
    :param expression_truth: If provided, use ground truth per cell type relative expression values,
    output from companion scRNA fine mapping.
    :param rng: Numpy random generator to use
    :return: data.DeconvolutionResult
    """
    if rng is None:
        rng = np.random.default_rng()

    # detetermine the number of spots
    n_nodes = reads.shape[0]

    # load the count matrix
    if n_gene is None:
        n_gene = reads.shape[1]
        Observation = reads
    elif isinstance(n_gene, (list, np.ndarray)):
        n_gene = len(n_gene)
        Observation = reads[:, n_gene]
    elif isinstance(n_gene, int):
        n_gene = min(n_gene, reads.shape[1])
        top = np.argsort(np.std(np.log(1 + reads), axis=0))[::-1]
        Observation = reads[:, top[:n_gene]]
    else:
        raise ValueError('n_gene must be a integer or a list of indices of genes')

    gfm = model_bkg.GraphFusedMultinomial(
        n_components=n_components,
        edges=edges,
        observations=Observation,
        n_gene=n_gene,
        lam_psi=lam2,
        background_noise=bkg,
        lda_initialization=lda,
        truth_expression=expression_truth,
        rng=rng)

    cell_prob_trace = np.zeros((n_samples, n_nodes, n_components + 1))
    cell_num_trace = np.zeros((n_samples, n_nodes, n_components + 1))
    expression_trace = np.zeros((n_samples, n_components, n_gene))
    beta_trace = np.zeros((n_samples, n_components))
    reads_trace = np.zeros((n_samples, n_nodes, n_gene, n_components))

    total_samples = n_samples * n_thin + n_burnin
    for step in range(total_samples):
        logger.info(f'Step {step}/{total_samples} ...')
        # perform Gibbs sampling
        gfm.sample(Observation)
        # save the trace of GFMM parameters
        if step >= n_burnin and (step - n_burnin) % n_thin == 0:
            idx = (step - n_burnin) // n_thin
            cell_prob_trace[idx] = gfm.probs
            expression_trace[idx] = gfm.phi
            beta_trace[idx] = gfm.beta
            cell_num_trace[idx] = gfm.cell_num
            reads_trace[idx] = gfm.reads

    return data.DeconvolutionResult(
        cell_prob_trace=cell_prob_trace,
        expression_trace=expression_trace,
        beta_trace=beta_trace,
        cell_num_trace=cell_num_trace,
        reads_trace=reads_trace,
        lam2=lam2,
        n_components=n_components
    )


class MarkerGeneMethod(Enum):
    TIGHT = 1
    FALSE_DISCOVERY_RATE = 2


def select_marker_genes(
        deconvolution_result: data.DeconvolutionResult,
        n_marker: int = 5,
        alpha: float = 0.05,
        method: MarkerGeneMethod = MarkerGeneMethod.TIGHT) -> List[np.ndarray]:
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
            if marker_idx_control.sum() < n_marker:
                raise RuntimeError(
                    'Less than {} genes satisfy omega > 1 - {}. Only {} genes satify the condition.'.format(
                        n_marker, alpha, marker_idx_control.sum()))
        elif method is MarkerGeneMethod.FALSE_DISCOVERY_RATE:
            sorted_index = np.argsort(1 - deconvolution_result.omega[k])
            fdr = np.cumsum(1 - deconvolution_result.omega[k][sorted_index]) / (np.arange(sorted_index.shape[0]) + 1)
            marker_control = np.argwhere(fdr <= alpha).flatten()
            marker_idx_control = sorted_index[marker_control]
        else:
            raise ValueError(method)
        # sort adjointly by omega_kg (primary) and expression level (secondary)
        top_marker = np.lexsort(
            (deconvolution_result.relative_expression[k][marker_idx_control],
             deconvolution_result.omega[k][marker_idx_control])
        )[::-1]

        if n_marker > len(top_marker):
            logger.warning(f'For cell type ({k}) fewer then ({n_marker}) genes '
                           f'met the marker gene criteria, will only use '
                           f'{len(top_marker)} marker genes for this cell type.')

        marker_gene_sets.append(marker_idx_control[top_marker[:n_marker]])

    return marker_gene_sets


def add_deconvolution_results_to_dataset(
        stdata: data.SpatialExpressionDataset,
        result: data.DeconvolutionResult):
    """
    Modify stdata in-place to annotate it with selected marker genes

    :param stdata: data.SpatialExpressionDataset to modify
    :param result: data.DeconvolutionResult to use
    """
    cell_num_matrix = result.cell_num_trace[:, :, 1:].mean(axis=0)
    cell_prob_matrix = result.cell_prob_trace[:, :, 1:].mean(axis=0)

    cell_prob_matrix_full = np.zeros((stdata.n_spot, cell_prob_matrix.shape[1]))

    cell_prob_matrix_full[stdata.tissue_mask] = cell_prob_matrix

    cell_num_matrix_full = np.zeros((stdata.n_spot, cell_num_matrix.shape[1]))

    cell_num_matrix_full[stdata.tissue_mask] = cell_num_matrix

    stdata.adata.obsm[data.CELL_TYPE_PROB_ATTR] = cell_prob_matrix_full
    stdata.adata.obsm[data.CELL_TYPE_COUNT_ATTR] = cell_num_matrix_full

    stdata.adata.uns[data.N_CELL_TYPES_ATTR] = result.n_components
    stdata.adata.varm[data.OMEGA_DIFFERENCE_ATTR] = result.omega_difference.T


def add_marker_gene_results_to_dataset(
        stdata: data.SpatialExpressionDataset,
        marker_genes: List[np.ndarray]):
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


def plot_cell_num(
        stdata: data.SpatialExpressionDataset,
        output_dir: str,
        output_format: str = 'pdf',
        cmap=cm.jet,
        seperate_pdf: bool = False,
        cell_type_names: Optional[List[str]] = None):
    plot_object = stdata.cell_type_counts

    if seperate_pdf:
        for i in range(stdata.n_cell_types):
            fig, ax = plt.subplot()

            if cell_type_names is not None:
                title = cell_type_names[i]
            else:
                title = f'Cell Type {i + 1}'

            ax.set_title(title)
            plotting.plot_colored_spatial_polygon(
                fig=fig,
                ax=ax,
                coords=stdata.positions_tissue,
                values=plot_object[:, i],
                layout=stdata.layout,
                colormap=cmap)
            ax.set_axis_off()

            fig.savefig(os.path.join(output_dir, f'cell_type_counts_{i}.{output_format}'))
            plt.close(fig)
    else:
        fig, axes = plt.subplots(ncols=stdata.n_cell_types,
                                 subplot_kw=dict(adjustable='box', aspect='equal'))
        fig.set_figwidth(fig.get_size_inches()[0] * stdata.n_cell_types)

        for i, ax in enumerate(axes):

            if cell_type_names is not None:
                title = cell_type_names[i]
            else:
                title = f'Cell Type {i + 1}'

            ax.set_title(title)
            plotting.plot_colored_spatial_polygon(
                fig=fig,
                ax=ax,
                coords=stdata.positions_tissue,
                values=plot_object[:, i],
                layout=stdata.layout,
                colormap=cmap
            )
            ax.set_axis_off()
        fig.savefig(os.path.join(output_dir, f'cell_type_counts.{output_format}'))
        plt.close(fig)


def plot_cell_prob(
        stdata: data.SpatialExpressionDataset,
        output_dir: str,
        output_format: str = 'pdf',
        cmap=cm.jet,
        seperate_pdf: bool = False,
        cell_type_names: Optional[List[str]] = None):
    plot_object = stdata.cell_type_probabilities

    if seperate_pdf:
        for i in range(stdata.n_cell_types):
            fig, ax = plt.subplot()
            if cell_type_names is not None:
                title = cell_type_names[i]
            else:
                title = f'Cell Type {i + 1}'

            ax.set_title(title)
            plotting.plot_colored_spatial_polygon(
                fig=fig,
                ax=ax,
                coords=stdata.positions_tissue,
                values=plot_object[:, i],
                layout=stdata.layout,
                colormap=cmap)
            ax.set_axis_off()

            fig.savefig(os.path.join(output_dir, f'cell_type_probability_{i}.{output_format}'))
            plt.close(fig)
    else:
        fig, axes = plt.subplots(
            ncols=stdata.n_cell_types,
            subplot_kw=dict(adjustable='box', aspect='equal'))

        fig.set_figwidth(fig.get_size_inches()[0] * stdata.n_cell_types)

        for i, ax in enumerate(axes):
            if cell_type_names is not None:
                title = cell_type_names[i]
            else:
                title = f'Cell Type {i + 1}'

            ax.set_title(title)
            plotting.plot_colored_spatial_polygon(
                fig=fig,
                ax=ax,
                coords=stdata.positions_tissue,
                values=plot_object[:, i],
                layout=stdata.layout,
                colormap=cmap
            )
            ax.set_axis_off()
        fig.savefig(os.path.join(output_dir, f'cell_type_probabilities.{output_format}'))
        plt.close(fig)


def plot_marker_genes(
        stdata: data.SpatialExpressionDataset,
        output_file: str,
        cell_type_labels: Optional[List[str]] = None,
        colormap: cm.ScalarMappable = cm.BuPu):
    all_gene_indices = np.concatenate(stdata.marker_gene_indices)
    all_gene_names = stdata.gene_names[all_gene_indices]
    n_marker = len(all_gene_indices)

    if cell_type_labels is None:
        cell_type_labels = ['Cell Type {}'.format(i + 1) for i in range(stdata.n_cell_types)]

    fig, (ax_genes, ax_legend) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [n_marker, 2]})
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
            s=norm(abs(stdata.omega_difference[k][all_gene_indices])) * inches_per_column * fig.dpi * 3,
            cmap=colormap,
            norm=norm)
        offset = offset + len(marker_gene_set)
    divider_lines.append(offset)

    ax_genes.set_xticks(np.arange(n_marker))
    ax_genes.set_xticklabels(all_gene_names, fontweight='bold', rotation=45, ha='right', va='top')
    ax_genes.set_yticks([x + 1 for x in range(stdata.n_cell_types)])
    ax_genes.set_yticklabels(cell_type_labels, rotation=0, fontweight='bold')
    ax_genes.invert_yaxis()
    ax_genes.margins(x=0.02, y=0.1)
    for x in divider_lines:
        ax_genes.axvline(x - 0.5, ls='--', c='k', alpha=0.5)

    legend_values = np.array([0.25, 0.5, 0.75, 1])
    norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    ax_legend.scatter(np.ones(len(legend_values)),
                      np.arange(len(legend_values)),
                      cmap=colormap,
                      c=legend_values,
                      s=legend_values * inches_per_column * fig.dpi * 3,
                      norm=norm)

    ax_legend.set_yticks(np.arange(len(legend_values)))
    ax_legend.set_yticklabels(['0.25', '0.5', '0.75', '1'], fontweight='bold')
    ax_legend.yaxis.tick_right()
    ax_legend.set_xticks([])
    ax_legend.spines["top"].set_visible(False)
    ax_legend.spines["right"].set_visible(False)
    ax_legend.spines["bottom"].set_visible(False)
    ax_legend.spines["left"].set_visible(False)
    ax_legend.set_ylabel('Loading', rotation=270, labelpad=50, fontweight='bold', fontsize='25')
    ax_legend.yaxis.set_label_position("right")
    ax_legend.margins(y=0.2)

    fig.tight_layout()
    fig.savefig(output_file, bbox_inches='tight')

    plt.close(fig)


def plot_cell_num_scatterpie(
        stdata: data.SpatialExpressionDataset,
        output_path: str,
        cell_type_names: Optional[List[str]] = None):
    """
    Create a "scatter pie" plot of the deconvolution cell counts.

    :param stdata: SpatialExpressionDataset to plot
    :param output_path: Where to save plot
    :param cell_type_names: Cell type names to use in plot, an array of length n_components
    """
    fig, ax = plt.subplots()

    plotting.plot_spatial_pie_charts(fig, ax,
                                     stdata.positions_tissue,
                                     values=stdata.cell_type_counts,
                                     layout=stdata.layout,
                                     plotting_coordinates=stdata.positions,
                                     cell_type_names=cell_type_names)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_deconvolution(stdata: data.SpatialExpressionDataset,
                       output_dir: str,
                       output_format: str = 'pdf',
                       cell_type_names: Optional[List[str]] = None):
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
        cell_type_names=cell_type_names)

    plot_cell_prob(
        stdata=stdata,
        output_dir=output_dir,
        output_format=output_format,
        seperate_pdf=False,
        cell_type_names=cell_type_names)

    plot_marker_genes(
        stdata=stdata,
        output_file=os.path.join(output_dir, f'marker_genes.{output_format}'),
        cell_type_labels=cell_type_names)

    plot_cell_num_scatterpie(
        stdata=stdata,
        output_path=os.path.join(output_dir, f'cell_num_scatterpie.{output_format}'),
        cell_type_names=cell_type_names)


def create_top_gene_lists(stdata: data.SpatialExpressionDataset,
                          deconvolution_result: data.DeconvolutionResult,
                          output_path: str,
                          n_marker_genes: int = 5,
                          alpha: float = 0.05,
                          marker_gene_method: MarkerGeneMethod = MarkerGeneMethod.TIGHT,
                          cell_type_names=None):
    marker_genes = select_marker_genes(
        deconvolution_result=deconvolution_result,
        n_marker=n_marker_genes,
        alpha=alpha,
        method=marker_gene_method)

    results = []
    for k, marker_gene_set in enumerate(marker_genes):
        result = pandas.DataFrame()
        result['gene_name'] = stdata.gene_names[marker_gene_set]

        result['rank_in_cell_type'] = np.arange(0, len(marker_gene_set))

        if cell_type_names is None:
            result['cell_type'] = np.repeat(np.array([k + 1]), n_marker_genes)
        else:
            result['cell_type'] = np.repeat(np.array(cell_type_names[k]), n_marker_genes)
        results.append(result)

    pandas.concat(results).to_csv(output_path, header=True, index=False)
