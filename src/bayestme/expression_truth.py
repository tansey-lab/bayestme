from typing import List

import numpy as np
import pandas
import pyro
import torch
from pyro import distributions as dist
from pyro.infer import MCMC, NUTS

from bayestme import data


def dirichlet_alpha_model(expression_truth=None, N=None, J=None, K=None):
    if expression_truth is not None:
        N = expression_truth.shape[0]
        K = expression_truth.shape[1]
        J = expression_truth.shape[2]

    with pyro.plate("J", J):
        with pyro.plate("K", K):
            alpha = pyro.sample("alpha", dist.Gamma(1, 1))

    with pyro.plate("N", N):
        sampled = pyro.sample(
            "obs", dist.Dirichlet(alpha).to_event(1), obs=expression_truth
        )

    return sampled


def fit_alpha_for_multiple_samples(data, num_warmup=200, num_samples=200):
    L = np.min(data[data > 0]) / 10.0
    data[data == 0] = L
    data = data / data.sum(axis=1)[:, None]

    tensor_data = torch.Tensor(data)

    mcmc_kernel = NUTS(dirichlet_alpha_model)
    mcmc = MCMC(mcmc_kernel, warmup_steps=num_warmup, num_samples=num_samples)
    mcmc.run(expression_truth=tensor_data)
    return mcmc.get_samples()["alpha"].mean(axis=0)


def combine_multiple_expression_truth(
    expression_truth_arrays: List[np.array], num_warmup=200, num_samples=200
):
    if len(expression_truth_arrays) < 2:
        return next(iter(expression_truth_arrays))

    arr = np.empty(
        (
            len(expression_truth_arrays),
            expression_truth_arrays[0].shape[0],
            expression_truth_arrays[0].shape[1],
        )
    )

    for index, sample in enumerate(expression_truth_arrays):
        arr[index, :, :] = sample

    result = fit_alpha_for_multiple_samples(
        arr.astype(float),
        num_warmup=num_warmup,
        num_samples=num_samples,
    )

    return result.detach().numpy()


def load_expression_truth(stdata: data.SpatialExpressionDataset, seurat_output: str):
    """
    Load outputs from seurat fine mapping to be used in deconvolution.

    :param stdata: SpatialExpressionDataset object
    :param seurat_output: CSV output from seurat fine mapping workflow
    :return: Tuple of n_components x n_genes size array, representing relative
    expression of each gene in each cell type
    """
    df = pandas.read_csv(seurat_output, index_col=0)

    phi_k_truth = df.loc[stdata.gene_names].to_numpy()

    L = np.min(phi_k_truth[phi_k_truth > 0]) / 10.0
    if L <= 0:
        L = 1e-9

    phi_k_truth[phi_k_truth == 0] = L

    # re-normalize so expression values sum to 1 within each component for
    # this subset of genes
    phi_k_truth_normalized = phi_k_truth / phi_k_truth.sum(axis=0)

    expression_truth = phi_k_truth_normalized.T

    return expression_truth
