import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.nn.parameter import Parameter
from typing import Optional
from bayestme.data import (
    SpatialExpressionDataset,
    DeconvolutionResult,
    SpatialDifferentialExpressionResult,
)
import tqdm


logger = logging.getLogger(__name__)


class SpatialPrograms(nn.Module):
    def __init__(
        self,
        log_rates,
        n_cell_types,
        n_programs,
        n_genes,
        n_spots,
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng()
        super().__init__()
        self.log_rates = log_rates
        self.n_cell_types = n_cell_types
        self.n_programs = n_programs
        self.n_genes = n_genes
        self.W = Parameter(
            torch.FloatTensor(
                rng.normal(0, 0.01, size=(n_cell_types, n_programs, n_spots))
            )
        )
        self.V = Parameter(
            torch.FloatTensor(
                rng.normal(0, 0.01, size=(n_cell_types, n_genes, n_programs))
            )
        )

    def forward(self, spots, genes):
        return (
            (
                self.log_rates[:, genes][..., spots]
                + (self.W[:, None, :, spots] * self.V[:, genes, :, None]).sum(dim=2)
            )
            .exp()
            .sum(dim=0)
        )


def edges_to_linear_tf(edges):
    # How many nodes are there?
    node_ids = np.unique(edges)

    # Get the neighbors for everyone and build the trend filter lists
    neighbors = {i: [] for i in range(len(node_ids))}
    tf1, tf2, tf3 = [], [], []
    for i, j in edges:
        for k in neighbors[i]:
            tf1.append(k)
            tf2.append(i)
            tf3.append(j)
        for k in neighbors[j]:
            tf1.append(k)
            tf2.append(j)
            tf3.append(i)
        neighbors[i].append(j)
        neighbors[j].append(i)

    return (
        np.array(tf1, dtype=int),
        np.array(tf2, dtype=int),
        np.array(tf3, dtype=int),
    )


def train(
    data: SpatialExpressionDataset,
    deconv: DeconvolutionResult,
    n_programs: int,
    n_steps=10000,
    batchsize_spots=1000,
    batchsize_genes=50,
    tf_lam=1,
    lasso_lam=0.1,
    rng: Optional[np.random.Generator] = None,
) -> SpatialDifferentialExpressionResult:
    if rng is None:
        rng = np.random.default_rng()
    r_flat = deconv.reads_trace.mean(axis=0)  # <N Spots> x <N Genes> x <N Cell Types>
    r_flat = r_flat.transpose(2, 1, 0)  # <N Cell Types> x <N Genes> x <N Spots>
    r_flat = np.clip(r_flat, 1e-10, np.inf)
    trendfilter_indices = edges_to_linear_tf(data.edges)
    n_spots = data.n_spot_in
    n_genes = data.n_gene

    t_log_rates = torch.tensor(np.log(r_flat))
    t_Reads = torch.tensor(data.counts)
    model = SpatialPrograms(
        log_rates=t_log_rates,
        n_cell_types=data.n_cell_types,
        n_programs=n_programs,
        n_genes=n_genes,
        n_spots=n_spots,
        rng=rng,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for step in tqdm.trange(n_steps):
        # Set the model to training mode
        model.train()

        # Reset the gradient
        model.zero_grad()

        # Sample some genes
        batch_genes = rng.choice(
            n_genes, size=min(n_genes, batchsize_genes), replace=False
        )

        # Sample some random spots
        batch_spots = rng.choice(
            n_spots, size=min(n_spots, batchsize_spots), replace=False
        )

        # Get the trend filtering penalties
        tf_mask = np.in1d(trendfilter_indices[1], batch_spots)

        # Tensor up
        batch_tf1 = torch.LongTensor(trendfilter_indices[0][tf_mask])
        batch_tf2 = torch.LongTensor(trendfilter_indices[1][tf_mask])
        batch_tf3 = torch.LongTensor(trendfilter_indices[2][tf_mask])
        batch_spots = torch.LongTensor(batch_spots)
        batch_genes = torch.LongTensor(batch_genes)

        # Get the spatial rates
        t_spatial_rates = model(batch_spots, batch_genes)

        # Get all the likelihoods
        t_log_like = (
            torch.distributions.Poisson(t_spatial_rates)
            .log_prob(t_Reads[batch_genes][:, batch_spots])
            .mean()
        )

        # Calculate the trend filter penalty
        t_trend_filter = (
            (
                model.W[..., batch_tf1]
                - 2 * model.W[..., batch_tf2]
                + model.W[..., batch_tf3]
            )
            .abs()
            .mean()
        )

        # Add a lasso regularizer to enforce V is sparse
        t_lasso = model.V[:, batch_genes].abs().mean()

        # Get the regularized loss
        loss = -t_log_like + tf_lam * t_trend_filter + lasso_lam * t_lasso

        # Calculate gradients
        loss.backward()

        # Apply the update
        optimizer.step()

    model.eval()
    w_hat = model.W.detach().numpy()  # (n_cell_types, n_programs, n_spots)
    v_hat = model.V.detach().numpy()  # (n_cell_types, n_genes, n_programs)

    return SpatialDifferentialExpressionResult(
        w_hat=w_hat,
        v_hat=v_hat,
    )
