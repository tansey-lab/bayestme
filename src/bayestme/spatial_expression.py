import numpy as np
import pypolyagamma
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.lines import Line2D
from scipy.sparse import block_diag, spdiags
from scipy.stats import nbinom
from matplotlib import colors
from scipy.stats import pearsonr
from libpysal.weights import W as pysal_Weights
from esda.moran import Moran

from bayestme.utils import ilogit, stable_softmax, sample_mvn_from_precision
from bayestme import utils, data, plotting

logger = logging.getLogger(__name__)


class SpatialDifferentialExpression:
    def __init__(self,
                 n_cell_types,
                 n_spatial_patterns,
                 Obs,
                 edges,
                 alpha_0=10,
                 prior_var=100.0,
                 lam2=1):
        # number of cell type from cell-typing results
        self.n_cell_types = n_cell_types
        # number of spots
        self.n_nodes = Obs.shape[0]
        # number of genes
        self.n_signals = Obs.shape[1]
        # number of spatial pattern per cell-type
        self.n_spatial_patterns = n_spatial_patterns
        self.lam2 = lam2

        # spatial patterns setup
        self.alpha = np.ones(self.n_spatial_patterns + 1)
        self.alpha[0] = alpha_0
        self.alpha[1:] = 1 / self.n_spatial_patterns

        # spatial pattern weights setup
        np.random.seed(0)
        self.W = np.zeros((self.n_cell_types, self.n_spatial_patterns + 1, self.n_nodes))
        self.Gamma = np.array([np.random.dirichlet(self.alpha) for _ in range(self.n_cell_types)])
        self.H = np.array(
            [np.random.choice(self.n_spatial_patterns + 1, p=g, size=(self.n_signals)) for g in self.Gamma]).T

        # Sample the spatial signal multipliers
        self.C = np.random.normal(0, 0.01, size=(self.n_signals, self.n_cell_types))
        self.V = np.random.normal(0, 0.01, size=(self.n_signals, self.n_cell_types))

        # Calculate the success probabilities
        Theta = np.array([[W_k[h] * v + c for h, v, c in zip(self.H[:, k], self.V[:, k], self.C[:, k])] for k, W_k in
                          enumerate(self.W)])
        self.Theta = np.transpose(Theta, [2, 1, 0])

        # PG variables
        self.pg = pypolyagamma.PyPolyaGamma(seed=42)
        self.Omegas = np.ones((self.n_signals, self.n_cell_types, self.n_nodes))
        self.prior_vars = np.repeat(prior_var, 2)
        D = utils.construct_edge_adjacency(edges)
        self.Delta = utils.construct_composite_trendfilter(D, 2, sparse=True)
        n_dims = self.n_spatial_patterns + 1
        self.DeltaT = block_diag([self.Delta.T for _ in range(n_dims)], format='csc')
        self.Delta = block_diag([self.Delta for _ in range(n_dims)], format='csc')
        self.Tau2, self.Tau2_c, self.Tau2_b, self.Tau2_a = utils.sample_horseshoe_plus(size=self.Delta.shape[0])
        self.Tau2 = self.Tau2.clip(0, 9)
        lam_Tau = spdiags(1 / (self.lam2 * self.Tau2), 0, self.Tau2.shape[0], self.Tau2.shape[0], format='csc')
        self.Sigma0_inv = self.DeltaT.dot(lam_Tau).dot(self.Delta)
        self.Cov_mats = np.zeros((self.W.shape[0], self.W.shape[1], self.n_nodes, self.n_nodes))
        for i in range(self.W.shape[0]):
            for j in range(1, self.W.shape[1]):
                self.Cov_mats[i, j] = self.Sigma0_inv[self.n_nodes * j:self.n_nodes * (j + 1),
                                      self.n_nodes * j:self.n_nodes * (j + 1)].todense()

    def sample_pg(self, rates, Y_igk):
        Theta_r = np.transpose(self.Theta, [1, 2, 0])
        Y_r = np.transpose(Y_igk, [1, 2, 0])
        rates_r = np.transpose(rates, [1, 2, 0])
        Trials = Y_r + rates_r
        obs_mask = Trials > 1e-4  # PG is very unstable for small trials; just ignore these as they give virtually no evidence anyway.
        obs_mask_flat = obs_mask.reshape(-1)
        trials_flat = Trials.reshape(-1)[obs_mask_flat].astype(float)
        thetas_flat = Theta_r.reshape(-1)[obs_mask_flat]
        omegas_flat = self.Omegas.reshape(-1)[obs_mask_flat]
        self.pg.pgdrawv(trials_flat, thetas_flat, omegas_flat)
        self.Omegas[obs_mask] = np.clip(omegas_flat, 1e-13, None)
        self.Omegas[~obs_mask] = 0.

    def sample_spatial_weights(self, n_obs, Y_igk):
        # Cache repeatedly calculated local variables
        Prior_precision = np.diag(1 / self.prior_vars)
        X = np.concatenate([np.ones(self.W.shape)[..., None], self.W[..., None]], axis=-1)

        for g in range(self.V.shape[0]):
            for k in range(self.V.shape[1]):
                # Calculate the precision term
                h = self.H[g, k]
                Precision = (X[k, h].T * self.Omegas[None, g, k]).dot(X[k, h]) + Prior_precision

                # Calculate the mean term
                mu_part = X[k, h].T.dot((Y_igk[:, g, k] - n_obs[:, g, k]) / 2)
                # Sample the offset and spatial weights
                c, v = sample_mvn_from_precision(Precision, mu_part=mu_part, sparse=False)
                self.C[g, k] = c
                self.V[g, k] = v

    def sample_spatial_patterns(self, n_obs, Y_igk, cell_type_filter):
        for k in range(self.W.shape[0]):
            for j in range(1, self.W.shape[1]):
                mask = self.H[:, k] == j
                if mask.sum() == 0:
                    # Pick a random instance to use as data rather than sampling from the prior
                    mask[np.random.choice(self.H.shape[0])] = True

                # for each cell-type, only look at the spots where contains sufficint number of cell from that 
                # cell-type (defined by cell_type_filter) to inference spatial pattern
                Y_masked = Y_igk[cell_type_filter[k]][:, mask, k].T
                n_obs_masked = n_obs[cell_type_filter[k]][:, mask, k].T
                Omegas_masked = self.Omegas[mask, k][:, cell_type_filter[k]]
                V_masked = self.V[mask, k:k + 1]
                C_masked = self.C[mask, k:k + 1]

                # PG likelihood terms
                a_j = (Omegas_masked * V_masked ** 2).sum(axis=0)
                b_j = (V_masked * ((Y_masked - n_obs_masked) / 2 - (Omegas_masked * C_masked))).sum(axis=0)

                # Posterior precision
                Precision = np.copy(self.Cov_mats[k, j][np.ix_(cell_type_filter[k], cell_type_filter[k])])
                Precision[np.diag_indices(Precision.shape[0])] += a_j

                # Posterior mu term
                mu_part = b_j

                # Sample the spatial pattern
                self.W[k, j, cell_type_filter[k]] = sample_mvn_from_precision(Precision, mu_part=mu_part, sparse=False)

    def sample_sigmainv(self, stability=1e-6, lam2=1):
        for i in range(self.W.shape[0]):
            deltas = self.Delta.dot(self.W[i].reshape(-1))
            rate = deltas ** 2 / (2 * lam2) + 1 / self.Tau2_c.clip(stability, 1 / stability)
            self.Tau2 = 1 / np.random.gamma(1, 1 / rate.clip(stability, 1 / stability)).clip(stability, 1 / stability)
            self.Tau2_c = 1 / np.random.gamma(1, 1 / (1 / self.Tau2 + 1 / self.Tau2_b).clip(stability, 1 / stability))
            self.Tau2_b = 1 / np.random.gamma(1, 1 / (1 / self.Tau2_c + 1 / self.Tau2_a).clip(stability, 1 / stability))
            self.Tau2_a = 1 / np.random.gamma(1, 1 / (1 / self.Tau2_b + 1).clip(stability, 1 / stability))
            lam_Tau = spdiags(1 / (lam2 * self.Tau2), 0, self.Tau2.shape[0], self.Tau2.shape[0], format='csc')
            self.Sigma0_inv = self.DeltaT.dot(lam_Tau).dot(self.Delta)
            for j in range(1, self.W.shape[1]):
                self.Cov_mats[i, j] = self.Sigma0_inv[self.W.shape[-1] * j:self.W.shape[-1] * (j + 1),
                                      self.W.shape[-1] * j:self.W.shape[-1] * (j + 1)].todense()

    def sample_spatial_assignments(self, n_obs, Y_igk):
        for g in range(self.n_signals):
            for k in range(self.n_cell_types):
                thetas = np.clip(ilogit(self.W[k] * self.V[g, k] + self.C[g, k]), 1e-6, 1 - 1e-6)
                logprobs = nbinom.logpmf(Y_igk[None, :, g, k], np.clip(n_obs[None, :, g, k], 1e-6, None), 1 - thetas)
                logprobs = logprobs.sum(axis=1)
                logprior = np.log(self.Gamma[k])
                p = stable_softmax(logprobs + logprior)
                self.H[g, k] = np.random.choice(self.n_spatial_patterns + 1, p=p)

    def sample_pattern_probs(self):
        # Conjugate Dirichlet update
        # print(H.shape, alpha.shape, H.max())
        for k, H_k in enumerate(self.H.T):
            self.Gamma[k] = np.random.dirichlet(
                np.array([(H_k == s).sum() for s in range(self.alpha.shape[0])]) + self.alpha)

    def sample(self, n_obs, Y_igk, cell_type_filter):
        self.sample_pg(n_obs, Y_igk)
        self.sample_spatial_patterns(n_obs, Y_igk, cell_type_filter)
        self.sample_sigmainv(stability=1e-6, lam2=self.lam2)
        self.sample_spatial_weights(n_obs, Y_igk)
        self.Theta = np.array(
            [[W_k[h] * v + c for h, v, c in zip(self.H[:, k], self.V[:, k], self.C[:, k])] for k, W_k in
             enumerate(self.W)])
        self.Theta = np.transpose(self.Theta, [2, 1, 0])
        self.sample_spatial_assignments(n_obs, Y_igk)
        self.sample_pattern_probs()

    def spatial_detection(self, cell_num_trace, beta_trace, expression_trace, reads_trace, n_samples=100, n_burn=100,
                          n_thin=5, ncell_min=5, simple=False):
        if len(cell_num_trace.shape) == 3:
            n_posterior_sample = cell_num_trace.shape[0]
        else:
            n_posterior_sample = 0
        self.W_samples = np.zeros((n_samples, self.n_cell_types, self.n_spatial_patterns + 1, self.n_nodes))
        self.C_samples = np.zeros((n_samples, self.n_signals, self.n_cell_types))
        self.Gamma_samples = np.zeros((n_samples, self.n_cell_types, self.n_spatial_patterns + 1))
        self.H_samples = np.zeros((n_samples, self.n_signals, self.n_cell_types), dtype=int)
        self.V_samples = np.zeros((n_samples, self.n_signals, self.n_cell_types))
        self.Theta_samples = np.zeros((n_samples, self.n_nodes, self.n_signals, self.n_cell_types))

        if n_posterior_sample > 0:
            cell_type_filter = (cell_num_trace[:, :, 1:].mean(axis=0) > ncell_min).T
            rate = np.array([beta_trace[i][:, None] * expression_trace[i] for i in range(n_posterior_sample)])
            reads = reads_trace.mean(axis=0).astype(int)
            lambdas = cell_num_trace.mean(axis=0)[:, 1:, None] * rate.mean(axis=0)[None]
        else:
            cell_type_filter = (cell_num_trace[:, 1:] > ncell_min).T
            rate = beta_trace[:, None] * expression_trace
            reads = reads_trace.astype(int)
            lambdas = cell_num_trace[:, 1:, None] * rate[None]

        for step in range(n_burn + n_samples):
            print(f'Step {step}')
            if step < n_burn:
                n_iter = 1
                Y_igk = reads
                n_obs_vector = np.transpose(lambdas, [0, 2, 1])
            else:
                n_iter = n_thin
                if n_posterior_sample == 0 or simple:
                    Y_igk = reads
                    n_obs_vector = np.transpose(lambdas, [0, 2, 1])
                else:
                    Y_igk = reads_trace[step - n_burn]
                    lambdas = cell_num_trace[step - n_burn, :, 1:, None] * rate[step - n_burn, None]
                    n_obs_vector = np.transpose(lambdas, [0, 2, 1])
            for i in range(n_iter):
                self.sample(n_obs_vector, Y_igk, cell_type_filter)
            if step >= n_burn:
                idx = step - n_burn
                self.W_samples[idx] = self.W
                self.C_samples[idx] = self.C
                self.Gamma_samples[idx] = self.Gamma
                self.H_samples[idx] = self.H
                self.V_samples[idx] = self.V
                self.Theta_samples[idx] = self.Theta


def run_spatial_expression(
        dataset: data.SpatialExpressionDataset,
        deconvolve_results: data.DeconvolutionResult,
        n_spatial_patterns: int,
        n_samples: int,
        n_burn: int,
        n_thin: int,
        n_cell_min: int,
        alpha0: int,
        prior_var: float,
        lam2: int,
        simple=True) -> data.SpatialDifferentialExpressionResult:
    sde = SpatialDifferentialExpression(
        n_cell_types=deconvolve_results.n_components,
        n_spatial_patterns=n_spatial_patterns,
        Obs=dataset.reads,
        edges=dataset.edges,
        alpha_0=alpha0,
        prior_var=prior_var,
        lam2=lam2
    )

    sde.spatial_detection(
        deconvolve_results.cell_num_trace,
        deconvolve_results.beta_trace,
        deconvolve_results.expression_trace,
        deconvolve_results.reads_trace,
        n_samples=n_samples,
        n_burn=n_burn,
        n_thin=n_thin,
        ncell_min=n_cell_min,
        simple=simple)

    return data.SpatialDifferentialExpressionResult(
        w_samples=sde.W_samples,
        c_samples=sde.C_samples,
        gamma_samples=sde.Gamma_samples,
        h_samples=sde.H_samples,
        v_samples=sde.V_samples,
        theta_samples=sde.Theta_samples
    )


def calculate_spatial_genes(sde_result: data.SpatialDifferentialExpressionResult,
                            h_threshold=0.95,
                            magnitude_filter=None) -> np.array:
    score = (sde_result.h_samples > 0).mean(axis=0)
    spatial_gene = []
    for gene, cell_type in np.argwhere(score > h_threshold):
        exp_gene = sde_result.theta_samples[:, :, gene, cell_type].mean(axis=0)
        if magnitude_filter:
            if exp_gene.max() - exp_gene.min() > magnitude_filter:
                spatial_gene.append([gene, cell_type])
        else:
            spatial_gene.append([gene, cell_type])
    return np.array(spatial_gene)


def moran_i(
        edges: np.ndarray,
        data: np.array,
        two_tailed=False) -> float:
    """
    Calculate Moran's I (spatial auto-correlation statistic)

    :param edges: N x 2 np.ndarray, where N is the number of edges.
                  Values in the edges matrix represent indices into data.
    :param data: np.array of scalar values
    :param two_tailed: If true, return 2-tailed p-value. Default is False.
    :return: float between 0 and 1
    """
    data = data[np.newaxis]

    neighbours = dict()
    weights = dict()
    for i in range(edges.max() + 1):
        nb = []
        for edge in edges[edges[:, 0] == i]:
            nb.append(edge[1])
        for edge in edges[edges[:, 1] == i]:
            nb.append(edge[0])
        neighbours[i] = nb
        weights[i] = [1] * len(nb)

    w = pysal_Weights(neighbours, weights)

    result = np.zeros(data.shape[:-1])
    for k in range(data.shape[0]):
        if len(result.shape) > 1:
            for h in range(data.shape[1]):
                result[k, h] = Moran(data[k, h, :], w, two_tailed=two_tailed).I
        else:
            result[k] = Moran(data[k, :], w, two_tailed=two_tailed).I

    return result[0]


def get_n_cell_correlation(n_cell_filter: np.array, w_pattern: np.array):
    """
    Return perason's R between two arrays
    
    :param n_cell_filter: np.array
    :param w_pattern: np.array
    :return: float
    """
    return pearsonr(n_cell_filter, w_pattern)[0]


def get_proportion_of_spots_in_k_with_pattern_h_per_gene(
        h_samples: np.ndarray,
        k: int,
        h: int) -> np.array:
    """
    
    :param h_samples: h_samples from SpatialDifferentialExpressionResult
    :param k: cell type index
    :param h: spatial pattern index
    :return: float
    """
    return (h_samples[:, :, k] == h).mean(axis=0)


PSEUDOGENE_MARKER = ':'


def filter_pseudogenes_from_selection(gene_id_selection: np.array, gene_names: np.array):
    """
    Given an array indexing genes from gene_names, remove
    any selection of a pseudogene.

    :param gene_id_selection: Array indexing into gene_names
    :param gene_names: Array of string, gene names
    :return:
    """
    mask = np.array([PSEUDOGENE_MARKER in g for g in gene_names[gene_id_selection].flatten()])
    return gene_id_selection[~mask]


def select_significant_spatial_programs(
        stdata: data.SpatialExpressionDataset,
        decon_result: data.DeconvolutionResult,
        sde_result: data.SpatialDifferentialExpressionResult,
        tissue_threshold: int = 5,
        cell_correlation_threshold: float = 0.5,
        moran_i_score_threshold: float = 0.9,
        gene_spatial_pattern_proportion_threshold: float = 0.95,
        filter_pseudogenes: bool = False):
    """
    Filter significant combinations of cell type and spatial expression patterns.
    This methodology aims to filter out programs that capture technical noise and
    over-dispersion rather than meaningful spatial signal.

    :param stdata: spatial expression dataset
    :param decon_result: deconvolution result
    :param sde_result: spatial differential expression result
    :param cell_correlation_threshold: Threshold for cell correlation metric
    :param moran_i_score_threshold: Moran's I score cutoff,
                                    spatial programs with scores below this will be filtered out
    :param gene_spatial_pattern_proportion_threshold: Only return (k, h) pairs where in cell types k
    greater than this proportion of spots are labeled with spatial pattern h for at least one gene.
    :param tissue_threshold: Only consider spots with greater than this many cells of type k
                             for Moran's I calculation and cell correlation calculation
    :param filter_pseudogenes: Do not consider pseudogenes.
    :return: generator of (cell type index, spatial pattern index, np.array of gene indices)
    """
    for k in range(sde_result.n_components):
        cell_number_mask = decon_result.cell_num_trace[:, :, k + 1].mean(axis=0) > tissue_threshold
        n_cells_of_type_k_per_spot = decon_result.cell_num_trace[:, :, k + 1].mean(axis=0)[cell_number_mask]
        pos_filter = stdata.positions_tissue[:, cell_number_mask].astype(int)
        edges_filter = utils.get_edges(pos_filter, stdata.layout.value)

        for h in range(1, sde_result.n_spatial_patterns + 1):
            gene_proportions_with_pattern_in_k = get_proportion_of_spots_in_k_with_pattern_h_per_gene(
                h_samples=sde_result.h_samples, h=h, k=k)
            gene_ids = np.argwhere(
                gene_proportions_with_pattern_in_k > gene_spatial_pattern_proportion_threshold)

            if len(gene_ids) == 0:
                logger.debug(
                    f'No genes have proportion > gene_spatial_pattern_proportion_threshold '
                    f'{gene_spatial_pattern_proportion_threshold} '
                    f'for cell type {k} pattern {h}, dropping.')
                continue

            w_pattern = sde_result.w_samples[:, k, h, :].mean(axis=0).copy()[cell_number_mask]

            if filter_pseudogenes:
                gene_ids = filter_pseudogenes_from_selection(gene_id_selection=gene_ids, gene_names=stdata.gene_names)

                if len(gene_ids) == 0:
                    logger.debug(
                        f'No non-pesudogenes have proportion > gene_spatial_pattern_proportion_threshold '
                        f'{gene_spatial_pattern_proportion_threshold} '
                        f'for cell type {k} pattern {h}, dropping.')
                    continue

            n_cell_correlation = get_n_cell_correlation(n_cells_of_type_k_per_spot, w_pattern)

            if abs(n_cell_correlation) >= cell_correlation_threshold:
                logger.debug(
                    f'n_cell_correlation >= cell_correlation_threshold '
                    f'{cell_correlation_threshold} '
                    f'for cell type {k} pattern {h}, dropping.')
                continue

            moran_i_score = moran_i(edges_filter, w_pattern)

            if moran_i_score <= moran_i_score_threshold:
                logger.debug(
                    f'Moran I score <= moran_i_score_threshold '
                    f'{moran_i_score_threshold} '
                    f'for cell type {k} pattern {h}, dropping.')
                continue

            logger.debug(
                f'Significant spatial pattern found: cell type {k}, pattern {h}.')
            yield k, h, gene_ids.flatten()


def plot_spatial_pattern_legend(
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        stdata: data.SpatialExpressionDataset,
        sde_result: data.SpatialDifferentialExpressionResult,
        gene_ids: np.array,
        k: int,
        colormap):
    loadings = sde_result.v_samples[:, gene_ids, k].mean(axis=0).flatten()
    loadings /= np.max(np.abs(loadings)) * np.sign(loadings[np.argmax(np.abs(loadings))])
    genes_selected = gene_ids[abs(loadings) > 0.1]
    loadings = loadings[abs(loadings) > 0.1]
    plot_order = np.argsort(loadings)
    loading_plot = loadings[plot_order]
    genes_selected_in_plot_order = genes_selected[plot_order]
    vmin = min(-1e-4, loading_plot.min())
    vmax = max(1e-4, loading_plot.max())
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    legend_elements = []
    for i, g in enumerate(genes_selected_in_plot_order):
        legend_elements.append(
            Line2D([0], [0], marker='o',
                   color=colormap(norm(loading_plot[i])),
                   label=stdata.gene_names[genes_selected_in_plot_order[i]],
                   markerfacecolor=colormap(norm(loading_plot[i])),
                   markersize=abs(loading_plot[i]) * 20,
                   linestyle='none')
        )
    legend_elements.reverse()

    ax.legend(handles=legend_elements, loc='center', fontsize=10, labelspacing=2, frameon=False)
    ax.set_axis_off()


def plot_spatial_pattern(
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        stdata: data.SpatialExpressionDataset,
        decon_result: data.DeconvolutionResult,
        sde_result: data.SpatialDifferentialExpressionResult,
        gene_ids: np.array,
        k: int,
        h: int,
        colormap,
        plot_threshold: int = 2):
    plot_mask = decon_result.cell_num_trace[:, :, k + 1].mean(axis=0) > plot_threshold
    loadings = sde_result.v_samples[:, gene_ids, k].mean(axis=0).flatten()
    rank = abs(loadings).argsort()[::-1]
    loadings = loadings[rank]
    w_plot = sde_result.w_samples[:, k, h, :].mean(axis=0).copy() * np.max(np.abs(loadings)) * np.sign(
        loadings[np.argmax(np.abs(loadings))])
    w_plot[~plot_mask] = 0

    vmin = min(-1e-4, w_plot[plot_mask].min())
    vmax = max(1e-4, w_plot[plot_mask].max())
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plotting.plot_colored_spatial_polygon(
        fig=fig,
        ax=ax,
        coords=stdata.positions_tissue,
        values=w_plot,
        layout=stdata.layout,
        norm=norm,
        colormap=colormap)
    ax.set_axis_off()


def plot_spatial_pattern_with_legend(
        stdata: data.SpatialExpressionDataset,
        decon_result: data.DeconvolutionResult,
        sde_result: data.SpatialDifferentialExpressionResult,
        gene_ids: np.array,
        k: int,
        h: int,
        output_file: str,
        colormap=cm.coolwarm,
        plot_threshold: int = 2):
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})
    plot_spatial_pattern_legend(
        fig=fig,
        ax=ax1,
        stdata=stdata,
        sde_result=sde_result,
        gene_ids=gene_ids,
        k=k,
        colormap=colormap
    )
    plot_spatial_pattern(
        fig=fig,
        ax=ax2,
        stdata=stdata,
        decon_result=decon_result,
        sde_result=sde_result,
        gene_ids=gene_ids,
        k=k,
        h=h,
        plot_threshold=plot_threshold,
        colormap=colormap
    )
    fig.savefig(output_file)
    plt.close(fig)


def plot_significant_spatial_patterns(
        stdata: data.SpatialExpressionDataset,
        decon_result: data.DeconvolutionResult,
        sde_result: data.SpatialDifferentialExpressionResult,
        output_dir,
        output_format: str = 'pdf'):
    significant_programs = select_significant_spatial_programs(
        stdata=stdata,
        decon_result=decon_result,
        sde_result=sde_result,
    )

    for k, h, gene_ids in significant_programs:
        plot_spatial_pattern_with_legend(
            stdata=stdata,
            decon_result=decon_result,
            sde_result=sde_result,
            gene_ids=gene_ids,
            k=k,
            h=h,
            output_file=os.path.join(output_dir, 'spatial_loading_cell_type_{}_{}.{}'.format(k, h, output_format))
        )
