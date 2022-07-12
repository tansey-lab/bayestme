import numpy as np
import pypolyagamma
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import block_diag, spdiags
from scipy.stats import nbinom
from matplotlib import colors
from scipy.stats import pearsonr
from libpysal.weights import W as pysal_Weights
from esda.moran import Moran

from .utils import ilogit, stable_softmax, sample_mvn_from_precision
from . import utils, data, plotting

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
        self.Delta = utils.composite_trendfilter(D, 2, sparse=True)
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
        data: np.ndarray,
        two_tailed=False):
    '''
    Calculate Moran's I (spatial auto-correlation statistic)
    N nodes, M edges
    edges: 2 by M
    data: 2d or 3d, K (by H) by N
    '''
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

    return result


def plot_spatial_patterns(
        stdata: data.SpatialExpressionDataset,
        decon_result: data.DeconvolutionResult,
        sde_result: data.SpatialDifferentialExpressionResult,
        output_dir,
        output_format: str = 'pdf',
        n_top: int = 10,
        tissue_threshold: int = 5,
        plot_threshold: int = 2,
        cell_correlation_threshold: float = 0.5,
        moran_i_score_threshold: float = 0.9,
        gene_spatial_pattern_proportion_threshold: float = 0.95):
    if stdata.layout is data.Layout.HEX:
        marker = 'H'
        size = 5
    else:
        marker = 's'
        size = 10

    for k in range(sde_result.n_components):
        cell_number_mask = decon_result.cell_num_trace[:, :, k + 1].mean(axis=0) > tissue_threshold
        plot_mask = decon_result.cell_num_trace[:, :, k + 1].mean(axis=0) > plot_threshold
        n_cell_filter = decon_result.cell_num_trace[:, :, k + 1].mean(axis=0)[cell_number_mask]
        pos_filter = stdata.positions_tissue[:, cell_number_mask].astype(int)
        edges_filter = utils.get_edges(pos_filter, stdata.layout.value)
        for h in range(1, sde_result.n_spatial_patterns + 1):
            gene_ids = np.argwhere(
                (sde_result.h_samples[:, :, k] == h).mean(axis=0) > gene_spatial_pattern_proportion_threshold)
            w_pattern = sde_result.w_samples[:, k, h, :].mean(axis=0).copy()[cell_number_mask]
            n_sp_gene = len(gene_ids)
            if n_sp_gene > 0:
                mask = np.array([':' in g for g in stdata.gene_names[gene_ids].flatten()])
                n_sp_gene = (~mask).sum()
            moran_i_score = moran_i(edges_filter, w_pattern[None])[0]
            n_cell_correlation = pearsonr(n_cell_filter, w_pattern)[0]
            if n_sp_gene != 0 and moran_i_score > moran_i_score_threshold and abs(
                    n_cell_correlation) < cell_correlation_threshold:
                loadings = sde_result.v_samples[:, gene_ids, k].mean(axis=0).flatten()
                rank = abs(loadings).argsort()[::-1]
                gene_ids = gene_ids.flatten()[rank]
                loadings = loadings[rank]
                w_plot = sde_result.w_samples[:, k, h, :].mean(axis=0).copy() * np.max(np.abs(loadings)) * np.sign(
                    loadings[np.argmax(np.abs(loadings))])
                w_plot[~plot_mask] = 0
                vmin = min(-1e-4, w_plot[plot_mask].min())
                vmax = max(1e-4, w_plot[plot_mask].max())
                norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                with sns.axes_style('white'):
                    plt.rc('font', weight='bold')
                    plt.rc('grid', lw=3)
                    plt.rc('lines', lw=1)
                    matplotlib.rcParams['pdf.fonttype'] = 42
                    matplotlib.rcParams['ps.fonttype'] = 42
                    subfigs = plotting.st_plot_with_room_for_legend(
                        data=w_plot[None, None],
                        pos=stdata.positions_tissue,
                        unit_dist=size,
                        cmap='coolwarm',
                        v_min=None,
                        v_max=None,
                        layout=marker,
                        subtitles=None, norm=norm,
                        x_y_swap=False, invert=[1, 1])

                loadings /= np.max(np.abs(loadings)) * np.sign(loadings[np.argmax(np.abs(loadings))])
                genes_selected = gene_ids[abs(loadings) > 0.1]
                loadings = loadings[abs(loadings) > 0.1]
                n_genes = len(genes_selected)
                if n_genes > n_top:
                    genes_selected = gene_ids[:n_top]
                    loadings = loadings[:n_top]
                    n_genes = n_top
                ax = subfigs[0].subplots(1, 1)
                plot_order = np.argsort(loadings)
                loading_plot = loadings[plot_order]
                gene_selected = genes_selected[plot_order]
                vmin = min(-1e-4, loading_plot.min())
                vmax = max(1e-4, loading_plot.max())
                norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                ax.scatter(
                    np.ones(n_genes),
                    np.arange(n_genes) + 1,
                    c=loading_plot,
                    s=abs(loading_plot) * 500,
                    cmap='coolwarm',
                    norm=norm)
                for g in range(n_genes):
                    ax.text(1 + 0.03, g + 1, stdata.gene_names[gene_selected[g]], ha='left', va='center')
                ax.set_yticks([])
                ax.set_xticks([])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.margins(x=0, y=0.15)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir,
                                 'spatial_loading_cell_type_{}_{}.{}'.format(k, h, output_format)),
                    bbox_inches='tight')
                plt.close()
