import numpy as np
from scipy.stats import binom

from bayestme import hmm_fast


def test_hmm_fast():
    n_components = 3

    # Di prob
    prob_in = [0.5, 0.5]

    # Truth cell type prob
    Truth_prior = np.array([[0.1, 0.6, 0.3], [0.1, 0.5, 0.4]])
    n_max = 120
    n_nodes = Truth_prior.shape[0]
    start_prob = np.array(
        [binom.pmf(np.arange(n_max + 1), n_max, p=p) for p in prob_in]
    )

    # Truth cell number [D_i, d_i1, d_i2, d_i3]
    Truth_n_cell = np.array([[60, 6, 37, 17], [62, 7, 30, 25]])

    # Truth r, 4 genes
    Truth_r = np.array([[5.5, 1, 5, 3], [2.1, 10, 10, 1.1], [4.4, 5, 3.3, 8]])

    lams = Truth_n_cell[:, 1:][:, :, None] * Truth_r[None]
    lams = np.clip(lams, 1e-6, None)
    rng = np.random.default_rng(0)

    n_sample = 1000
    n_trials = 3
    n_gene = 4
    means = np.zeros((n_trials, 2, Truth_n_cell.shape[1]))

    prob = np.zeros((n_nodes, n_components - 1))
    for k in range(n_components - 1):
        prob[:, k] = Truth_prior[:, k] / Truth_prior[:, k:].sum(axis=1)
    Transition = hmm_fast.transition_mat_vec(prob, n_max + 1)

    for trial in range(n_trials):
        print(trial)
        # p = np.ones((n_components, n_gene)) * 0.7
        Obs = rng.poisson(lams)
        print(Obs)

        hmm = hmm_fast.HMM(n_components, 120)

        cell_sample = np.zeros((1000, 2, n_components + 1))
        # Emission = lambda Obs, lams: nbinom.logpmf(Obs, lams, p=np.clip(1-p[-1], 1e-20, None))
        for i in range(n_sample):
            cell_sample[i] = hmm.ffbs(
                Obs,
                np.log(start_prob),
                LogTransition=np.log(Transition),
                expression=Truth_r,
            )
        means[trial] = cell_sample.mean(axis=0)
