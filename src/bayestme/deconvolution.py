import logging
import numpy as np
import matplotlib
import seaborn as sns
import os.path

from enum import Enum
from typing import Optional, List
from matplotlib import pyplot as plt
from matplotlib import colors
from bayestme import model_bkg, data, bayestme_plot

logger = logging.getLogger(__name__)


def deconvolve(
        reads,
        edges,
        n_gene=None,
        n_components=None,
        lam2=None,
        n_samples=100,
        n_burnin=1000,
        n_thin=10,
        random_seed=0,
        bkg=False,
        lda=False) -> data.DeconvolutionResult:
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

    np.random.seed(random_seed)

    gfm = model_bkg.GraphFusedMultinomial(
        n_components=n_components,
        edges=edges,
        observations=Observation,
        n_gene=n_gene,
        lam_psi=lam2,
        background_noise=bkg,
        lda_initialization=lda)

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


def detect_marker_genes(
        deconvolution_result: data.DeconvolutionResult,
        n_marker: int = 5,
        alpha: float = 0.05,
        method: MarkerGeneMethod = MarkerGeneMethod.TIGHT):
    """
    Returns (marker_gene, omega_difference)

    marker_gene: matrix of shape N components x N marker genes,
    where the values are indices of the marker genes
    in the gene name array.

    omega_difference: matrix of shape N components x N genes
    representing the average expression value / max expression value
    for each gene within each component.
    """

    gene_expression = deconvolution_result.expression_trace.mean(axis=0)
    n_components = gene_expression.shape[0]
    # omega size K by G
    omega = np.zeros_like(gene_expression)
    difference = np.zeros_like(gene_expression)
    expression = np.zeros_like(gene_expression)
    marker_gene = np.zeros((n_components, n_marker)).astype(int)
    for k in range(n_components):
        # max expression of each gene for each posterior sample
        max_exp = deconvolution_result.expression_trace.max(axis=1)
        omega[k] = (deconvolution_result.expression_trace[:, k] == max_exp).mean(axis=0)
        difference[k] = (deconvolution_result.expression_trace[:, k] / max_exp).mean(axis=0)
        mask = np.arange(n_components) != k
        max_exp_g = gene_expression[mask].max(axis=0)
        expression[k] = (gene_expression[k] - max_exp_g) / np.max(np.vstack([gene_expression, max_exp_g]), axis=0)
        # fdr control
        if method is MarkerGeneMethod.TIGHT:
            marker_control = omega[k] > 1 - alpha
            marker_idx_control = np.argwhere(marker_control).flatten()
            if marker_idx_control.sum() < n_marker:
                raise RuntimeError(
                    'Less than {} genes satisfy omega > 1 - {}. Only {} genes satify the condition.'.format(
                        n_marker, alpha, marker_idx_control.sum()))
        elif method is MarkerGeneMethod.FALSE_DISCOVERY_RATE:
            sorted_index = np.argsort(1 - omega[k])
            fdr = np.cumsum(1 - omega[k][sorted_index]) / (np.arange(sorted_index.shape[0]) + 1)
            marker_control = np.argwhere(fdr <= alpha).flatten()
            marker_idx_control = sorted_index[marker_control]
        else:
            raise ValueError(method)
        # sort adjointly by omega_kg (primary) and expression level (secondary)
        top_marker = np.lexsort((expression[k][marker_idx_control], omega[k][marker_idx_control]))[::-1]
        marker_gene[k] = marker_idx_control[top_marker[:n_marker]]

    return marker_gene, difference


def plot_cell_num(
        stdata: data.SpatialExpressionDataset,
        result: data.DeconvolutionResult,
        output_dir: str,
        cmap: str = 'jet',
        seperate_pdf: bool = False):
    if stdata.layout == data.Layout.HEX:
        layout = 'H'
        size = 5
    else:
        layout = 's'
        size = 10

    plot_object = result.cell_num_trace[:, :, 1:].mean(axis=0)
    if seperate_pdf:
        for i in range(result.n_components):
            bayestme_plot.st_plot(
                data=plot_object[:, i].T[:, np.newaxis],
                pos=stdata.positions_tissue,
                name='cell_number_component_{}'.format(i),
                save=output_dir,
                layout=layout,
                unit_dist=size,
                cmap=cmap)
    else:
        bayestme_plot.st_plot(
            data=plot_object.T[:, np.newaxis],
            pos=stdata.positions_tissue,
            name='cell_number',
            subtitles=['Cell type {}'.format(i+1) for i in range(result.n_components)],
            save=output_dir,
            layout=layout,
            unit_dist=size,
            cmap=cmap)


def plot_cell_prob(
        stdata: data.SpatialExpressionDataset,
        result: data.DeconvolutionResult,
        output_dir: str,
        cmap: str = 'jet',
        seperate_pdf: bool = False):
    if stdata.layout == data.Layout.HEX:
        layout = 'H'
        size = 5
    else:
        layout = 's'
        size = 10

    plot_object = result.cell_prob_trace[:, :, 1:].mean(axis=0)
    if seperate_pdf:
        for i in range(result.n_components):
            bayestme_plot.st_plot(
                data=plot_object[:, i].T[:, np.newaxis],
                pos=stdata.positions_tissue,
                name='cell_probability_component_{}'.format(i),
                save=output_dir,
                layout=layout,
                unit_dist=size,
                cmap=cmap)
    else:
        bayestme_plot.st_plot(
            data=plot_object.T[:, np.newaxis],
            pos=stdata.positions_tissue,
            name='cell_probability',
            subtitles=['Component {}'.format(i) for i in range(result.n_components)],
            save=output_dir,
            layout=layout,
            unit_dist=size,
            cmap=cmap)


def plot_marker_genes(
        marker_gene: np.ndarray,
        difference: np.ndarray,
        stdata: data.SpatialExpressionDataset,
        deconvolution_results: data.DeconvolutionResult,
        output_file: str,
        cell_type_labels: Optional[List[str]] = None):
    n_marker = marker_gene.flatten().shape[0]
    n_marker_genes_per_component = marker_gene.shape[1]
    marker_gene_names = stdata.gene_names[marker_gene.flatten()]

    if cell_type_labels is None:
        cell_type_labels = ['Cell Type {}'.format(i + 1) for i in range(deconvolution_results.n_components)]

    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=1)
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        fig = plt.figure(figsize=(n_marker + 1, deconvolution_results.n_components + 1))
        stbox = [0, 0, 0.87, 1]
        scbox = [0.878, 0, 0.015, 1]
        stframe = plt.axes(stbox)
        scframe = plt.axes(scbox)
        for k in range(deconvolution_results.n_components):
            vmin = min(-1e-4, difference[k][marker_gene.flatten()].min())
            vmax = max(1e-4, difference[k][marker_gene.flatten()].max())
            norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            img = stframe.scatter(np.arange(n_marker), np.ones(n_marker) * (k + 1),
                                  c=difference[k][marker_gene.flatten()],
                                  s=abs(difference[k][marker_gene.flatten()]) * 1200, cmap='BuPu',
                                  norm=norm)
        scatter_plot = np.array([0.25, 0.5, 0.75, 1])
        norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
        scframe.scatter(np.ones(4), scatter_plot * 4, cmap='BuPu', c=scatter_plot, s=scatter_plot * 1200, norm=norm)
        for i in range(deconvolution_results.n_components):
            stframe.axvline((i + 1) * n_marker_genes_per_component - 0.5, ls='--', c='k', alpha=0.5)
        scframe.set_yticks([1, 2, 3, 4, 5])
        scframe.set_yticklabels(['0.25', '0.5', '0.75', '1', ''], fontsize=35)
        scframe.yaxis.tick_right()
        scframe.set_xticks([])
        scframe.spines["top"].set_visible(False)
        scframe.spines["right"].set_visible(False)
        scframe.spines["bottom"].set_visible(False)
        scframe.spines["left"].set_visible(False)
        scframe.set_ylabel('Loading', fontsize=45, rotation=270, labelpad=48, fontweight='bold')
        scframe.yaxis.set_label_position("right")
        scframe.margins(y=0.2)
        stframe.set_xticks(np.arange(marker_gene.flatten().shape[0]) + 0.2)
        stframe.set_xticklabels(marker_gene_names, fontsize=40, fontweight='bold', rotation=45, ha='right', va='top')
        stframe.set_yticks([x + 1 for x in range(deconvolution_results.n_components)])
        stframe.set_yticklabels(cell_type_labels, fontsize=45, rotation=0, fontweight='bold')
        stframe.invert_yaxis()
        stframe.margins(x=0.02, y=0.1)
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()


def plot_deconvolution(stdata: data.SpatialExpressionDataset,
                       deconvolution_result: data.DeconvolutionResult,
                       output_dir: str,
                       n_marker_genes: int = 5,
                       alpha: float = 0.05,
                       marker_gene_method: MarkerGeneMethod = MarkerGeneMethod.TIGHT):
    plot_cell_num(
        stdata=stdata,
        result=deconvolution_result,
        output_dir=output_dir,
        seperate_pdf=False)
    plot_cell_prob(
        stdata=stdata,
        result=deconvolution_result,
        output_dir=output_dir,
        seperate_pdf=False)

    marker_genes, omega_difference = detect_marker_genes(
        deconvolution_result=deconvolution_result,
        n_marker=n_marker_genes,
        alpha=alpha,
        method=marker_gene_method)

    plot_marker_genes(
        marker_gene=marker_genes,
        difference=omega_difference,
        deconvolution_results=deconvolution_result,
        stdata=stdata,
        output_file=os.path.join(output_dir, 'marker_genes.pdf'))
