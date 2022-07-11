import logging
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import KFold
from typing import Iterable
from scipy.stats import multinomial

from bayestme import utils, data, plotting
from bayestme.model_bkg import GraphFusedMultinomial

logger = logging.getLogger(__name__)


def get_n_neighbors(stdata: data.SpatialExpressionDataset):
    """
    Return a numpy array of size stdata.n_spot_in representing the number of neighbors for each spot.
    """
    n_neighbours = np.zeros(stdata.n_spot_in)
    edges = utils.get_edges(stdata.positions_tissue, layout=stdata.layout.value)

    for i in range(stdata.n_spot_in):
        n_neighbours[i] = (edges[:, 0] == i).sum() + (edges[:, 1] == i).sum()

    return n_neighbours


def create_folds(stdata: data.SpatialExpressionDataset,
                 n_fold=5,
                 n_splits=15):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    n_neighbours = get_n_neighbors(stdata)

    if stdata.layout is data.Layout.HEX:
        edge_threshold = 5
    else:
        edge_threshold = 3

    splits = kf.split(np.arange(stdata.n_spot_in)[n_neighbours > edge_threshold])

    for k in range(n_fold):
        _, heldout = next(splits)
        mask = np.array(
            [i in np.arange(stdata.n_spot_in)[n_neighbours > edge_threshold][heldout] for i in range(stdata.n_spot_in)])
        yield mask


def plot_folds(stdata, folds, output_dir: str):
    n_neighbours = get_n_neighbors(stdata)
    fig, ax = plt.subplots(1, len(folds), figsize=(6 * (len(folds) + 1), 6))
    if len(folds) == 1:
        ax = [ax]

    for k, (lam, n_components, mask, fold_number) in enumerate(folds):
        plotting.plot_spots(ax[k], n_neighbours, stdata.positions_tissue, s=5, cmap='viridis')
        ax[k].scatter(stdata.positions_tissue[0, mask], stdata.positions_tissue[1, mask], s=5, c='r')
    plt.savefig(os.path.join(output_dir, 'k_fold_masks.pdf'))
    plt.close()


def get_phenotype_selection_parameters_for_folds(stdata: data.SpatialExpressionDataset,
                                                 n_fold: int,
                                                 n_splits: int,
                                                 lams: Iterable[int],
                                                 n_components_min: int,
                                                 n_components_max: int):
    for lam in lams:
        for n_components in range(n_components_min, n_components_max + 1):
            for fold_number, mask in enumerate(create_folds(stdata, n_fold, n_splits)):
                yield lam, n_components, mask, fold_number


def sample_graph_fused_multinomial(
        train: np.ndarray,
        test: np.ndarray,
        n_components,
        edges: np.ndarray,
        n_gene: int,
        lam_psi: float,
        background_noise: bool,
        lda_initialization: bool,
        mask: np.ndarray,
        n_max: int,
        n_samples: int,
        n_thin: int,
        n_burn: int):
    n_nodes = train.shape[0]

    heldout_spots = np.argwhere(mask).flatten()
    train_spots = np.argwhere(~mask).flatten()
    if len(heldout_spots) == 0:
        mask = None

    graph_fused_multinomial = GraphFusedMultinomial(
        n_components=n_components,
        edges=edges,
        observations=train,
        n_gene=n_gene,
        lam_psi=lam_psi,
        background_noise=background_noise,
        lda_initialization=lda_initialization,
        mask=mask,
        n_max=n_max)

    cell_prob_trace = np.zeros((n_samples, n_nodes, n_components + 1))
    cell_num_trace = np.zeros((n_samples, n_nodes, n_components + 1))
    expression_trace = np.zeros((n_samples, n_components, n_gene))
    beta_trace = np.zeros((n_samples, n_components))
    loglhtest_trace = np.zeros(n_samples)
    loglhtrain_trace = np.zeros(n_samples)

    for step in range(n_samples * n_thin + n_burn):
        if step % 10 == 0:
            logger.info(f'Step {step}')
        # perform Gibbs sampling
        graph_fused_multinomial.sample(train)
        # save the trace of GFMM parameters
        if step >= n_burn and (step - n_burn) % n_thin == 0:
            idx = (step - n_burn) // n_thin
            cell_prob_trace[idx] = graph_fused_multinomial.probs
            expression_trace[idx] = graph_fused_multinomial.phi
            beta_trace[idx] = graph_fused_multinomial.beta
            cell_num_trace[idx] = graph_fused_multinomial.cell_num
            rates = (graph_fused_multinomial.probs[:, 1:][:, :, None] * (
                    graph_fused_multinomial.beta[:, None] * graph_fused_multinomial.phi)[None])
            nb_probs = rates.sum(axis=1) / rates.sum(axis=(1, 2))[:, None]
            loglhtest_trace[idx] = np.array(
                [multinomial.logpmf(test[i], test[i].sum(), nb_probs[i]) for i in heldout_spots]).sum()
            loglhtrain_trace[idx] = np.array(
                [multinomial.logpmf(train[i], train[i].sum(), nb_probs[i]) for i in train_spots]).sum()
            logger.info('{}, {}'.format(loglhtrain_trace[idx], loglhtest_trace[idx]))

    return (
        cell_prob_trace,
        cell_num_trace,
        expression_trace,
        beta_trace,
        loglhtest_trace,
        loglhtrain_trace
    )


def run_phenotype_selection_single_job(
        lam: float,
        n_components: int,
        mask: np.ndarray,
        fold_number: int,
        stdata: data.SpatialExpressionDataset,
        n_samples: int,
        n_burn: int,
        n_thin: int,
        max_ncell: int,
        n_gene: int,
        background_noise: bool,
        lda_initialization: bool) -> data.PhenotypeSelectionResult:
    train = stdata.reads.copy()
    test = stdata.reads.copy()
    train[mask] = 0
    test[~mask] = 0

    n_gene = min(n_gene, stdata.n_gene)

    stddev_ordering = utils.get_stddev_ordering(train)
    train = train[:, stddev_ordering[:n_gene]]
    test = test[:, stddev_ordering[:n_gene]]

    (
        cell_prob_trace,
        cell_num_trace,
        expression_trace,
        beta_trace,
        loglhtest_trace,
        loglhtrain_trace
    ) = sample_graph_fused_multinomial(
        train=train,
        test=test,
        n_components=n_components,
        edges=stdata.edges,
        n_gene=n_gene,
        lam_psi=lam,
        background_noise=background_noise,
        lda_initialization=lda_initialization,
        mask=mask,
        n_max=max_ncell,
        n_samples=n_samples,
        n_thin=n_thin,
        n_burn=n_burn)

    return data.PhenotypeSelectionResult(
        mask=mask,
        cell_prob_trace=cell_prob_trace,
        expression_trace=expression_trace,
        beta_trace=beta_trace,
        cell_num_trace=cell_num_trace,
        log_lh_train_trace=loglhtrain_trace,
        log_lh_test_trace=loglhtest_trace,
        lam=lam,
        n_components=n_components,
        fold_number=fold_number)
