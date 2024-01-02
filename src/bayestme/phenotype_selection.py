import logging
import os
from typing import Iterable
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import multinomial
from sklearn.model_selection import KFold

import bayestme.common
from bayestme import data
from bayestme import deconvolution
from bayestme.common import InferenceType
from bayestme.plot import common

logger = logging.getLogger(__name__)


def get_n_neighbors(stdata: data.SpatialExpressionDataset):
    """
    Return a numpy array of size stdata.n_spot_in representing the number of neighbors for each spot.
    """
    n_neighbours = np.zeros(stdata.n_spot_in)
    edges = stdata.edges

    for i in range(stdata.n_spot_in):
        n_neighbours[i] = (edges[:, 0] == i).sum() + (edges[:, 1] == i).sum()

    return n_neighbours


def create_folds(
    stdata: data.SpatialExpressionDataset, n_fold=5, n_splits=15, n_neighbours=None
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    if n_neighbours is None:
        n_neighbours = get_n_neighbors(stdata)

    if stdata.layout is bayestme.common.Layout.HEX:
        edge_threshold = 5
    else:
        edge_threshold = 3

    splits = kf.split(np.arange(stdata.n_spot_in)[n_neighbours > edge_threshold])

    for k in range(n_fold):
        _, heldout = next(splits)
        mask = np.array(
            [
                i in np.arange(stdata.n_spot_in)[n_neighbours > edge_threshold][heldout]
                for i in range(stdata.n_spot_in)
            ]
        )
        yield mask


def plot_folds(stdata, folds, output_dir: str):
    fig, ax = plt.subplots(1, len(folds), figsize=(6 * (len(folds) + 1), 6))
    if len(folds) == 1:
        ax = [ax]

    patches = []
    for v, label in [(0, "Heldout"), (1, "Not Heldout")]:
        patches.append(Patch(color=common.Glasbey30(v), label=label))

    for k, mask in enumerate(folds):
        _, cb, _, _, _ = common.plot_colored_spatial_polygon(
            fig=fig,
            ax=ax[k],
            coords=stdata.positions_tissue.T,
            values=(~mask).astype(int),
            normalize=False,
            colormap=common.Glasbey30,
            plotting_coordinates=stdata.positions,
            layout=stdata.layout,
        )

        cb.remove()

        ax[k].set_axis_off()

        ax[k].set_title(f"Split {k + 1}")

    ax[k].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    plt.savefig(os.path.join(output_dir, "k_fold_masks.pdf"))
    plt.close()


def get_phenotype_selection_parameters_for_folds(
    stdata: data.SpatialExpressionDataset,
    n_fold: int,
    n_splits: int,
    lams: Iterable[int],
    n_components_min: int,
    n_components_max: int,
):
    n_neighbours = get_n_neighbors(stdata)

    for lam in lams:
        for n_components in range(n_components_min, n_components_max + 1):
            for fold_number, mask in enumerate(
                create_folds(stdata, n_fold, n_splits, n_neighbours)
            ):
                yield lam, n_components, mask, fold_number


def run_phenotype_selection_single_job(
    spatial_smoothing_parameter: float,
    n_components: int,
    mask: np.ndarray,
    fold_number: int,
    stdata: data.SpatialExpressionDataset,
    n_samples: int,
    mcmc_n_burn: int,
    mcmc_n_thin: int,
    n_svi_steps: int,
    background_noise: bool,
    lda_initialization: bool,
    use_spatial_guide: bool,
    inference_type: InferenceType = InferenceType.MCMC,
    rng: Optional[np.random.Generator] = None,
) -> data.PhenotypeSelectionResult:
    stdata_holdout = stdata.copy()
    stdata_holdout.counts[mask, :] = 0

    deconvolution_samples = deconvolution.sample_from_posterior(
        data=stdata_holdout,
        n_components=n_components,
        spatial_smoothing_parameter=spatial_smoothing_parameter,
        n_samples=n_samples,
        inference_type=inference_type,
        mcmc_n_burn=mcmc_n_burn,
        mcmc_n_thin=mcmc_n_thin,
        n_svi_steps=n_svi_steps,
        background_noise=background_noise,
        lda_initialization=lda_initialization,
        use_spatial_guide=use_spatial_guide,
        rng=rng,
    )

    nb_probs = deconvolution_samples.nb_probs

    heldout_counts = stdata.counts[None, mask, :]
    train_counts = stdata.counts[None, ~mask, :]

    nb_probs_holdout = nb_probs[:, mask, :]
    nb_probs_train = nb_probs[:, ~mask, :]

    loglhtest_trace = multinomial.logpmf(
        heldout_counts, heldout_counts.sum(axis=-1), nb_probs_holdout
    ).sum(axis=-1)
    loglhtrain_trace = multinomial.logpmf(
        train_counts, train_counts.sum(axis=-1), nb_probs_train
    ).sum(axis=-1)

    return data.PhenotypeSelectionResult(
        mask=mask,
        cell_prob_trace=deconvolution_samples.cell_prob_trace,
        expression_trace=deconvolution_samples.expression_trace,
        beta_trace=deconvolution_samples.beta_trace,
        cell_num_trace=deconvolution_samples.cell_num_trace,
        log_lh_train_trace=loglhtrain_trace,
        log_lh_test_trace=loglhtest_trace,
        lam=spatial_smoothing_parameter,
        n_components=n_components,
        fold_number=fold_number,
    )
