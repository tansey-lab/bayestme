import logging
from typing import Optional

import numpy as np
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from bayestme import data
from bayestme.mcmc import model_bkg

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
    bkg=False,
    lda=False,
    n_max=120,
    D=30,
    expression_truth=None,
    rng: Optional[np.random.Generator] = None,
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
        if expression_truth is not None:
            expression_truth = expression_truth[:, n_gene]
    elif isinstance(n_gene, int):
        n_gene = min(n_gene, reads.shape[1])
        top = np.argsort(np.std(np.log(1 + reads), axis=0))[::-1]
        Observation = reads[:, top[:n_gene]]
        if expression_truth is not None:
            expression_truth = expression_truth[:, top[:n_gene]]
    else:
        raise ValueError("n_gene must be a integer or a list of indices of genes")

    gfm = model_bkg.GraphFusedMultinomial(
        n_components=n_components,
        edges=edges,
        observations=Observation,
        n_gene=n_gene,
        lam_psi=lam2,
        background_noise=bkg,
        lda_initialization=lda,
        truth_expression=expression_truth,
        n_max=n_max,
        D=D,
        rng=rng,
    )

    cell_prob_trace = np.zeros((n_samples, n_nodes, n_components + 1))
    cell_num_trace = np.zeros((n_samples, n_nodes, n_components + 1))
    expression_trace = np.zeros((n_samples, n_components, n_gene))
    beta_trace = np.zeros((n_samples, n_components))
    reads_trace = np.zeros((n_samples, n_nodes, n_gene, n_components))

    total_samples = n_samples * n_thin + n_burnin
    with logging_redirect_tqdm():
        for step in tqdm.trange(total_samples, desc="Deconvolution"):
            logger.info(f"Step {step}/{total_samples} ...")
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
        cell_prob_trace=cell_prob_trace[:, :, 1:],
        expression_trace=expression_trace,
        beta_trace=beta_trace,
        cell_num_trace=cell_num_trace[:, :, 1:],
        reads_trace=reads_trace,
        lam2=lam2,
        n_components=n_components,
    )
