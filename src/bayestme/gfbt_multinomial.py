import numpy as np
from pypolyagamma import PyPolyaGamma
from scipy.sparse import spdiags, kron, csc_matrix

from . import utils


class GraphFusedBinomialTree:
    def __init__(self, n_classes, edges, trend_order=0, lam2=1e-5, pg_seed=42, stability=1e-6):
        self.n_classes = n_classes
        self.edges = edges
        self.trend_order = trend_order

        # Setup the random sampler for the Polya-Gamma latent variables
        self.pg = PyPolyaGamma(seed=pg_seed)

        # Initialize the graph trend filtering matrix
        self.D = utils.construct_edge_adjacency(self.edges)
        self.Delta = utils.construct_trendfilter(self.D, self.trend_order)
        self.n_nodes = self.D.shape[1]  # Number of trees being fused (vertices in the graph)

        # Initialize the nuisance variables for the Horseshoe+ prior
        self.lam2 = lam2  # TODO: put a prior on this
        self.Tau2, self.Tau2_c, self.Tau2_b, self.Tau2_a = utils.sample_horseshoe_plus(size=self.Delta.shape[0])
        self.Tau2 = self.Tau2.clip(0, 9)

        # Initialize the logits (thetas) and PG latent variables (omegas)
        self.Thetas = np.zeros((self.n_nodes, self.n_classes - 1))
        self.Omegas = np.zeros((self.n_nodes, self.n_classes - 1))

        #         # Create the class-to-path lookup tables
        #         self.paths = np.unpackbits(np.arange(self.n_classes, dtype=np.uint8)).reshape(self.n_classes, 8)
        #         self.paths = self.paths[:,self.paths.shape[1]-int(np.log2(self.n_classes)):].astype(int) # Reshape and squeeze the array
        #         depth_sizes = 2**np.arange(self.paths.shape[1]) # 1 2 4
        #         self.path_nodes = np.zeros(self.paths.shape, dtype=int)
        #         self.path_nodes[:,1:] = np.array([np.sum(self.paths[:,:i]*(depth_sizes[None,:i][:,::-1]),axis=1) for i in range(1,self.paths.shape[1])]).T
        #         self.path_nodes += (depth_sizes-1)[None]

        # Numerical stability parameter to deal with crazy horseshoe sampling tails
        self.stability = stability

        # Initialize the label probabilities
        self.calculate_probs()

    def counts_from_outcomes(self, outcomes):
        count_obs = np.cumsum(outcomes[:, ::-1], axis=1)[:, ::-1]
        Trials = count_obs[:, :-1].astype(float)
        Successes = count_obs[:, 1:].astype(float)

        return Trials, Successes

    def resample(self, outcomes):
        # If outcomes is an N x K array where N is the nodes and K is the classes,
        # it should contain the count for how many times each class was seen.
        Trials, Successes = self.counts_from_outcomes(outcomes)

        # Do we have any observations for this binomial tree node?
        obs_mask = Trials > 0
        obs_mask_flat = obs_mask.reshape(-1)

        # Sample all the polya-gamma latent variables
        with np.errstate(divide='ignore'):
            trials_flat = Trials.reshape(-1)[obs_mask_flat]
            thetas_flat = self.Thetas.reshape(-1)[obs_mask_flat]
            omegas_flat = self.Omegas.reshape(-1)[obs_mask_flat]
            self.pg.pgdrawv(trials_flat, thetas_flat, omegas_flat)
            self.Omegas[obs_mask] = omegas_flat
            self.Omegas[~obs_mask] = 0.

        # Construct the sparse precision matrix
        # The matrix is a block-banded matrix with band width dependent on the order of trend filtering.
        # Each block is n_nodes x n_nodes and the diagonal is composed of n_classes - 1 blocks.
        # The first block corresponds to the root nodes in the all the trees in the graph.
        # The last block corresponds to the bottom-right node in the trees.
        self.lam_Tau = spdiags(1 / (self.lam2 * self.Tau2), 0, self.Tau2.shape[0], self.Tau2.shape[0], format='csc')
        self.Sigma0_inv = kron(np.eye(self.n_classes - 1),
                               self.Delta.T.tocsc().dot(self.lam_Tau).dot(self.Delta)).tocsc()

        # The PG latent variables enter into the diagonal of the precision matrix. This handles that using
        # sparse arrays that respect the block-diagonal structure.
        diagonals = (np.arange(self.n_nodes)[None] + self.n_nodes * np.arange(self.n_classes - 1)[:, None]).flatten()
        self.Omega = csc_matrix((self.Omegas.T.reshape(-1), (diagonals, diagonals)), shape=self.Sigma0_inv.shape)

        # Sample the latent probability logits
        n, m = self.Sigma0_inv.shape
        ind = np.arange(n)
        correction = csc_matrix((np.ones(n) * 1e-6, (ind, ind)))
        self.Sigma_inv = self.Sigma0_inv + self.Omega + correction
        self.mu = np.zeros(self.Sigma_inv.shape[0])
        self.mu[diagonals] = Successes.T.reshape(-1) - Trials.T.reshape(-1) / 2
        self.Thetas = utils.sample_mvn_from_precision(self.Sigma_inv, mu_part=self.mu, sparse=True,
                                                      Q_shape=self.Sigma_inv.shape).reshape(self.Thetas.shape[1],
                                                                                            self.Thetas.shape[0]).T

        # Sample Horseshoe+ prior parameters
        deltas = self.Delta.dot(self.Thetas)
        rate = (deltas ** 2).sum(axis=1) / (2 * self.lam2) + 1 / self.Tau2_c.clip(self.stability, 1 / self.stability)
        self.Tau2 = 1 / np.random.gamma(self.n_classes / 2, 1 / rate.clip(self.stability, 1 / self.stability))
        self.Tau2_c = 1 / np.random.gamma(1, 1 / (1 / self.Tau2 + 1 / self.Tau2_b).clip(self.stability,
                                                                                        1 / self.stability))
        self.Tau2_b = 1 / np.random.gamma(1, 1 / (1 / self.Tau2_c + 1 / self.Tau2_a).clip(self.stability,
                                                                                          1 / self.stability))
        self.Tau2_a = 1 / np.random.gamma(1, 1 / (1 / self.Tau2_b + 1).clip(self.stability, 1 / self.stability))

        # Convert the binomial trees into class label probabilities
        self.calculate_probs()

    def calculate_probs(self):
        '''Converts from a binomial tree representation to a multinomial representation.'''
        split_probs = utils.ilogit(self.Thetas)
        prior = np.hstack((np.ones(self.n_nodes)[:, None], split_probs)).cumprod(axis=1)
        decision = 1 - np.hstack((split_probs, np.zeros(self.n_nodes)[:, None]))
        self.probs = prior * decision
