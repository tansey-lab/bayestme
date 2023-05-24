from typing import List

import numpy as np
import pandas

import pyro
import torch
from pyro import distributions as dist
from pyro.infer import MCMC, NUTS

from bayestme import data


def dirichlet_alpha_model(expression_truth, N=None, J=None):
    if expression_truth is not None:
        N = expression_truth.shape[0]
        J = expression_truth.shape[1]
    alpha = pyro.sample("alpha", dist.Gamma(torch.ones(J) * 0.1, torch.ones(J) * 0.1))

    with pyro.plate("N", N):
        pyro.sample("obs", dist.Dirichlet(alpha), obs=expression_truth)


def fit_alpha_for_multiple_samples(data, num_warmup=200, num_samples=200):
    L = np.min(data[data > 0]) / 10.0
    data[data == 0] = L
    data = data / data.sum(axis=1)[:, None]
    data = torch.tensor(data)
    mcmc = MCMC(
        NUTS(dirichlet_alpha_model), warmup_steps=num_warmup, num_samples=num_samples
    )

    mcmc.run(data)

    return mcmc.get_samples()["alpha"].mean(axis=0)


def combine_multiple_expression_truth(
    expression_truth_arrays: List[np.array], num_warmup=200, num_samples=200
):
    if len(expression_truth_arrays) < 2:
        return next(iter(expression_truth_arrays))

    cell_types = np.arange(expression_truth_arrays[0].shape[0])

    per_celltype_alpha_parameters = []

    for cell_type_idx in cell_types:
        cell_type_specific_counts = np.vstack(
            [arr[cell_type_idx, :] for arr in expression_truth_arrays]
        )
        per_celltype_alpha_parameters.append(
            fit_alpha_for_multiple_samples(
                cell_type_specific_counts.astype(float),
                num_warmup=num_warmup,
                num_samples=num_samples,
            )
        )

    return np.vstack(per_celltype_alpha_parameters)


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

    # re-normalize so expression values sum to 1 within each component for
    # this subset of genes
    phi_k_truth_normalized = phi_k_truth / phi_k_truth.sum(axis=0)

    return phi_k_truth_normalized.T
