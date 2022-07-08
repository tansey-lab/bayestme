import numpy as np
from scipy.stats import binom
import scipy.special as scs

from . import utils


def transition_mat_vec(phi, n_max, ifsigma=False):
    # get the binomial transition matrix
    T = np.zeros((phi.shape[0], phi.shape[1], n_max, n_max))
    if ifsigma:
        p = utils.ilogit(phi)
    else:
        p = phi
    p_s = np.power.outer(p, np.arange(n_max))
    p_f = np.power.outer(1 - p, np.arange(n_max))
    for n in range(n_max):
        coeff = np.array([scs.binom(n, k) for k in range(n + 1)])
        T[:, :, n, :(n + 1)] = coeff * p_s[:, :, :(n + 1)] * p_f[:, :, :(n + 1)][:, :, ::-1]
    return T


def emission_fast(reads, log_cell, expression, cell_grid):
    # calculate the emission prob of a given cell type k
    # reads:        N by G
    # log_cell:     n_max   
    # expression:   G
    # prob:         N by n_max
    # better clip the cell_grid at 0 before taking the log, in order to solve the log(0) problem
    prob = reads.sum(axis=1)[:, None] * log_cell - expression.sum() * cell_grid
    prob = prob - prob.max(axis=1)[:, None]
    # print(reads.shape)
    # print((cell_grid[:, None]*expression).shape)
    # prob = poisson.logpmf(reads[:, None], (cell_grid[:, None]*expression)[None]).sum(axis=-1)
    return prob


class HMM:
    def __init__(self, n_components, n_max=120):
        self.n_states = n_components
        self.n_max = n_max
        # cache the list of all possible cell numbers [0, 1, ..., n_max]
        self.cell_grid = np.arange(self.n_max + 1)
        with np.errstate(divide='ignore'):
            self.log_cell = np.clip(np.log(self.cell_grid), -1e30, None)
        # cache the idx arrays for ship_up function
        self.lower_idx = np.zeros((int((1 + n_max) * (2 + n_max) / 2), 2)).astype(int)
        self.upper_idx = np.zeros_like(self.lower_idx)
        idx = 0
        for i in range(1 + n_max):
            for j in range(i + 1):
                self.lower_idx[idx] = np.array([i, j])
                self.upper_idx[idx] = np.array([i - j, j])
                idx += 1

    def ship_up(self, mat):
        # transform a lower triangular matrix to right-upper triangular matrix
        out = np.ones_like(mat) * -np.inf
        out[:, self.upper_idx[:, 0], self.upper_idx[:, 1]] = mat[:, self.lower_idx[:, 0], self.lower_idx[:, 1]]
        return out

    def forward_filter(self, Obs):
        ### forward filtering
        ### calculate alpha(l_k) = P(d_ik, n_k, R_i1:k)
        ### Obs:            N by G by K
        ### predictor_log:  N by n_max by n_max
        ### start_prob:     N by n_max
        ### LogTrans:       N by k_components-1 by n_max by n_max
        ### lams:           n_max by G
        ### alpha_log:      N by K-1 by n_max (n_ik) by n_max (d_ik)
        for i in range(self.n_states - 1):
            # calculate \sum_{l_{k-1}} P(l_k|l_{k-1})\alpha(l_{k-1})
            if i == 0:
                # eqn 11
                predictor_log = self.start_prob[:, :, None] + self.LogTrans[:, i]
            else:
                # eqn 10
                predictor_log = scs.logsumexp(self.alpha_log[:, i - 1], axis=-1)[:, :, None] + self.LogTrans[:, i]
            # P(R_ik|d_ik)
            self.alpha_log[:, i] = self.ship_up(predictor_log) + emission_fast(Obs[:, i], self.log_cell,
                                                                               self.expression[i], self.cell_grid)[:,
                                                                 None]
        self.alpha_log[:, -1] += emission_fast(Obs[:, -1], self.log_cell, self.expression[-1], self.cell_grid)[:, :,
                                 None]

    def backward_sample(self):
        ### backward sampling
        ### calculate posterior P(d_ik|l_{k+1}, R_i,1:K)
        ### and sample d_ik from the posterior
        # d_ik sample
        samples = np.zeros((self.n_nodes, self.n_states + 1)).astype(int)
        # n_k samples
        samples_n = np.zeros((self.n_nodes, self.n_states + 1)).astype(int)
        for i in range(self.n_states):
            if i == 0:
                # eqn 13
                post_prob = scs.logsumexp(self.alpha_log[:, -1], axis=-1)
            else:
                # eqn 12
                post_prob = self.alpha_log[:, -i][np.arange(self.n_nodes), samples_n[:, i]]
            # normalize the posterior prob and sample
            raw_prob = post_prob.copy()
            peak = post_prob.max(axis=1)
            post_prob -= peak[:, None]
            post_prob = np.exp(post_prob)
            for j in range(self.n_nodes):
                cell_limit = self.n_max + 1 - samples_n[j, i]
                post_prob[j, cell_limit:] = 0
            zero_idx = np.argwhere(post_prob.sum(axis=1) == 0)
            # print(zero_idx)
            # print(len(zero_idx))
            # if len(zero_idx) > 0:
            #     print(raw_prob[post_prob])
            post_prob /= post_prob.sum(axis=1)[:, None]
            samples[:, i] = (post_prob.cumsum(axis=1) > np.random.rand(post_prob.shape[0])[:, None]).argmax(axis=1)
            samples_n[:, i + 1] = samples_n[:, i] + samples[:, i]
        # the last entry in samples_n is D_i, move it to the output vector 
        samples[:, -1] = samples_n[:, -1]
        self.samples = samples[:, ::-1]

    def ffbs(self, Obs, start_prob, LogTransition, expression):
        # both LogTransition and start_prob are logp
        # LogTrans:     K-1 by n_max by n_max
        # cell_count:   N by K
        # expression:   K by G
        # alpha_log:    N by K-1 by n_max by n_max
        self.LogTrans = LogTransition
        self.start_prob = start_prob
        self.expression = expression
        self.n_nodes = Obs.shape[0]
        self.alpha_log = np.zeros((self.n_nodes, self.n_states - 1, self.n_max + 1, self.n_max + 1))
        self.forward_filter(Obs)
        self.backward_sample()
        return self.samples


if __name__ == '__main__':
    n_components = 3

    # Di prob
    prob_in = [0.5, 0.5]

    # Truth cell type prob
    Truth_prior = np.array([[0.1, 0.6, 0.3],
                            [0.1, 0.5, 0.4]])
    n_max = 120
    n_nodes = Truth_prior.shape[0]
    start_prob = np.array([binom.pmf(np.arange(n_max + 1), n_max, p=p) for p in prob_in])

    # Truth cell number [D_i, d_i1, d_i2, d_i3]
    Truth_n_cell = np.array([[60, 6, 37, 17],
                             [62, 7, 30, 25]])

    # Truth r, 4 genes
    Truth_r = np.array([[5.5, 1, 5, 3],
                        [2.1, 10, 10, 1.1],
                        [4.4, 5, 3.3, 8]])

    lams = Truth_n_cell[:, 1:][:, :, None] * Truth_r[None]
    lams = np.clip(lams, 1e-6, None)
    rng = np.random.default_rng(0)

    n_sample = 1000
    n_trials = 100
    n_gene = 4
    means = np.zeros((n_trials, 2, Truth_n_cell.shape[1]))

    prob = np.zeros((n_nodes, n_components - 1))
    for k in range(n_components - 1):
        prob[:, k] = Truth_prior[:, k] / Truth_prior[:, k:].sum(axis=1)
    # print(prob)
    Transition = transition_mat_vec(prob, n_max + 1)

    for trial in range(n_trials):
        print(trial)
        # p = np.ones((n_components, n_gene)) * 0.7
        Obs = rng.poisson(lams)
        print(Obs)

        hmm = HMM(n_components, 120)

        cell_sample = np.zeros((1000, 2, n_components + 1))
        # Emission = lambda Obs, lams: nbinom.logpmf(Obs, lams, p=np.clip(1-p[-1], 1e-20, None))
        for i in range(n_sample):
            cell_sample[i] = hmm.ffbs(Obs, np.log(start_prob), LogTransition=np.log(Transition), expression=Truth_r)
        means[trial] = cell_sample.mean(axis=0)
        print(means[trial])
        # exit()
    import matplotlib.pyplot as plt

    fig, axarr = plt.subplots(2, Truth_n_cell.shape[1], figsize=(Truth_n_cell.shape[0] * 5, 5))
    for i in range(Truth_n_cell.shape[0]):
        for j in range(Truth_n_cell.shape[1]):
            axarr[i, j].hist(means[:, i, j], bins=np.linspace(means[:, i, j].min() - 1, means[:, i, j].max() + 1, 100))
            axarr[i, j].axvline(Truth_n_cell[i, j], color='black', lw=2, label='Truth')
            axarr[i, j].axvline(means[:, i, j].mean(), color='orange', lw=2, label='Posterior mean')
    plt.show()
