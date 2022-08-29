import numpy as np
import os.path
from typing import Optional

from scipy.stats import binom
from sklearn.decomposition import LatentDirichletAllocation

from bayestme import utils
from bayestme.hmm_fast import HMM, transition_mat_vec
from bayestme.gfbt_multinomial import GraphFusedBinomialTree


def transition_mat(phi, n_max, coeff, ifsigma=False):
    # get the binomial transition matrix
    T = np.zeros((n_max, n_max))
    if ifsigma:
        p = utils.ilogit(phi)
    else:
        p = phi
    p_s = p ** np.arange(n_max)
    p_f = (1 - p) ** np.arange(n_max)
    for n in range(n_max):
        T[n, :(n + 1)] = coeff[n] * p_s[:(n + 1)] * p_f[:(n + 1)][::-1]
    return T


class GraphFusedMultinomial:
    def __init__(self, n_components, edges, observations,
                 n_gene=300,
                 n_max=120,
                 background_noise=False,
                 mask=None,
                 c=4,
                 D=30,
                 lam_psi=1e-2,
                 lda_initialization=False,
                 truth_expression=None,
                 truth_prob=None,
                 truth_cellnum=None,
                 truth_reads=None,
                 truth_beta=None,
                 rng: Optional[np.random.Generator] = None):
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.n_components = n_components
        self.n_max = n_max
        self.n_gene = n_gene
        self.edges = edges
        self.bkg = background_noise
        self.HMM = HMM(self.n_components, self.n_max, rng=self.rng)
        self.gtf_psi = GraphFusedBinomialTree(self.n_components + 1, edges, lam2=lam_psi, rng=self.rng)
        self.mask = mask
        self.n_nodes = self.gtf_psi.n_nodes

        # initialize cell-type probs
        if truth_prob is not None:
            self.probs = truth_prob
        else:
            self.probs = np.ones(self.gtf_psi.probs.shape) * 1 / self.n_components
            self.probs[:, 0] = 0.5

        # initialize gene expression profile
        self.alpha = np.ones(self.n_gene)
        if truth_expression is not None:
            self.phi = truth_expression
        elif lda_initialization:
            print('Initializing with lda')
            lda = LatentDirichletAllocation(n_components=self.n_components, random_state=self.rng)
            lda.fit(observations)
            self.phi = lda.components_ / lda.components_.sum(axis=1)[:, None]
            self.probs[:, 1:] = lda.transform(observations)
        else:
            self.phi = self.rng.dirichlet(self.alpha, size=n_components)
        if self.bkg:
            bkg = np.ones(self.n_gene) / self.n_gene
            self.phi = np.vstack((self.phi, bkg))

        if self.bkg:
            self.cell_num = np.zeros((self.n_nodes, self.n_components + 2)).astype(int)
        else:
            self.cell_num = np.zeros((self.n_nodes, self.n_components + 1)).astype(int)
        if truth_cellnum is not None:
            self.cell_num[:, :-1] = truth_cellnum
            self.cell_num[:, -1] = 1
        else:
            self.cell_num[:, 0] = self.rng.binomial(self.n_max, self.probs[:, 0])
            if self.bkg:
                self.cell_num[:, -1] = 1
                self.cell_num[:, 1:-1] = utils.multinomial_rvs(self.cell_num[:, 0], p=self.probs[:, 1:], rng=self.rng)
            else:
                self.cell_num[:, 1:] = utils.multinomial_rvs(self.cell_num[:, 0], p=self.probs[:, 1:], rng=self.rng)

        if mask is not None:
            spot_count = observations[~mask].sum(axis=1)
        else:
            spot_count = observations.sum(axis=1)
        mu = spot_count.sum() / (self.n_nodes * D)
        L = np.percentile(spot_count, 5) / D
        U = np.percentile(spot_count, 95) / D
        s = max(mu - L, U - mu)
        self.a_beta = c ** 2 * mu ** 2 / s ** 2
        self.b_beta = c ** 2 * mu / s ** 2
        if truth_beta is not None:
            self.beta = truth_beta
        else:
            self.beta = self.rng.gamma(self.a_beta, 1 / self.b_beta, size=n_components)
        if self.bkg:
            self.beta = np.concatenate([self.beta, [np.min([observations.sum(axis=1).min(), 100])]])

        if truth_reads is not None:
            self.reads = truth_reads

        # get the transition matrices
        self.Transition = np.zeros((self.n_nodes, self.n_components - 1, self.n_max + 1, self.n_max + 1))
        self.expression = self.beta[:, None] * self.phi

    def sample_reads(self, Observations):
        '''
        sample cell-type-wise reads of each gene at each spot, R_igk
        reads:            N*G*K   R_igk
        Observation:      N*G     R_ig      observed data, gene reads at each spot
        assignment_probs: N*G*K   xi_igk    multinational prob
        cell_num:         N*K     d_ik      cell-type-wise cell count in each spot, d_ik, N*K
        betas:            K       beta_k    expected cell-type-wise total gene expression of individual cells
        '''
        self.expression = self.beta[:, None] * self.phi
        expected_counts = self.cell_num[:, 1:, None] * self.expression[None]
        self.assignment_probs = expected_counts / np.clip(expected_counts.sum(axis=1, keepdims=True), 1e-20, None)
        self.assignment_probs = np.transpose(self.assignment_probs, [0, 2, 1])
        # multinomial draw for all spots
        self.reads = utils.multinomial_rvs(Observations.astype(np.int64), self.assignment_probs, rng=self.rng)

    def sample_phi(self):
        '''
        sample cell-type-wise gene expression profile, phi_kg
        '''
        phi_posteriors = self.alpha[None] + self.reads.sum(axis=0).T
        self.phi = np.array([self.rng.dirichlet(c) for c in phi_posteriors])

    def sample_cell_num(self):
        '''
        sample the cell-type-wise cell count 
        cell_num[i] = [D_i, d_i1, d_i2, ..., d_iK], where d_i1 + d_i2 + ... + d_iK = D_i
        D_i     total cell number in spot i
        d_ik    cell number of cell-type k in spot i
        '''
        with np.errstate(divide='ignore'):
            prob = 1 - utils.ilogit(self.gtf_psi.Thetas[:, 1:])
            # print(np.argwhere(np.isinf(np.exp(-self.gtf_psi.Thetas[:, 1:]))))
            start_prob = np.array(
                [binom.logpmf(np.arange(self.n_max + 1), self.n_max, p=self.probs[i, 0]) for i in range(self.n_nodes)])
            self.Transition = transition_mat_vec(prob, self.n_max + 1)
            log_transition = np.log(self.Transition)
            if self.bkg:
                self.cell_num[:, :-1] = self.HMM.ffbs(np.transpose(self.reads[:, :, :-1], [0, 2, 1]), start_prob,
                                                      LogTransition=log_transition,
                                                      expression=self.expression[:-1])
            else:
                self.cell_num = self.HMM.ffbs(np.transpose(self.reads, [0, 2, 1]), start_prob,
                                              LogTransition=log_transition, expression=self.expression)

    def sample_probs(self):
        '''
        sample cell-type probability psi_ik with spatial smoothing
        '''
        # clean up the GFTB input cell num
        if self.bkg:
            cell_num = self.cell_num[:, :-1].copy()
        else:
            cell_num = self.cell_num.copy()
        cell_num[:, 0] = self.n_max - cell_num[:, 0]
        # GFTB sampling
        if self.mask is not None:
            cell_num[self.mask] = 0
        self.gtf_psi.resample(cell_num)
        # clean up the cell-type prob
        self.probs = self.gtf_psi.probs
        self.probs[:, 0] = 1 - self.probs[:, 0]
        self.probs[:, 1:] /= self.probs[:, 1:].sum(axis=1, keepdims=True)

    def sample_beta(self):
        '''
        sample expected cell-type-wise total cellular gene expression
        '''
        R_k = self.reads.sum(axis=(0, 1))
        d_k = self.cell_num[:, 1:].sum(axis=0)
        self.beta = self.rng.gamma(R_k + self.a_beta, 1 / (d_k + self.b_beta))

    def sample(self, Obs):
        self.sample_reads(Obs)
        self.sample_phi()
        self.sample_cell_num()
        self.sample_probs()
        self.sample_beta()

    def load_model(self, load_dir=''):
        # load model parameters
        self.cell_num = np.load(os.path.join(load_dir, 'cell_num.npy'))
        self.beta = np.load(os.path.join(load_dir, 'beta.npy'))
        self.phi = np.load(os.path.join(load_dir, 'phi.npy'))
        self.probs = np.load(os.path.join(load_dir, 'probs.npy'))
        self.reads = np.load(os.path.join(load_dir, 'reads.npy'))
        # load spatial smoothing parameters
        self.gtf_psi.Thetas = np.load(os.path.join(load_dir, 'checkpoint_Thetas'))
        self.gtf_psi.Omegas = np.load(os.path.join(load_dir, 'checkpoint_Omegas'))
        self.gtf_psi.Tau2 = np.load(os.path.join(load_dir, 'checkpoint_Tau2'))
        self.gtf_psi.Tau2_a = np.load(os.path.join(load_dir, 'checkpoint_Tau2_a'))
        self.gtf_psi.Tau2_b = np.load(os.path.join(load_dir, 'checkpoint_Tau2_b'))
        self.gtf_psi.Tau2_c = np.load(os.path.join(load_dir, 'checkpoint_Tau2_c'))

    def save_model(self, save_dir=''):
        np.save(os.path.join(save_dir, 'checkpoint_cell_num.npy'), self.cell_num)
        np.save(os.path.join(save_dir, 'checkpoint_beta.npy'), self.beta)
        np.save(os.path.join(save_dir, 'checkpoint_phi.npy'), self.phi)
        np.save(os.path.join(save_dir, 'checkpoint_probs.npy'), self.probs)
        np.save(os.path.join(save_dir, 'checkpoint_reads.npy'), self.reads)
        np.save(os.path.join(save_dir, 'checkpoint_Thetas'), self.gtf_psi.Thetas)
        np.save(os.path.join(save_dir, 'checkpoint_Omegas'), self.gtf_psi.Omegas)
        np.save(os.path.join(save_dir, 'checkpoint_Tau2'), self.gtf_psi.Tau2)
        np.save(os.path.join(save_dir, 'checkpoint_Tau2_a'), self.gtf_psi.Tau2_a)
        np.save(os.path.join(save_dir, 'checkpoint_Tau2_b'), self.gtf_psi.Tau2_b)
        np.save(os.path.join(save_dir, 'checkpoint_Tau2_c'), self.gtf_psi.Tau2_c)
