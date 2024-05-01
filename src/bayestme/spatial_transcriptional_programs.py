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
from bayestme.plot.common import plot_colored_spatial_polygon
import tqdm
import os.path
from matplotlib import pyplot as plt
from matplotlib import gridspec, cm
from matplotlib.colors import TwoSlopeNorm

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
    deconvolution_result: DeconvolutionResult,
    n_programs: int,
    n_steps=10000,
    batchsize_spots=1000,
    batchsize_genes=50,
    trend_filtering_lambda=1,
    lasso_lambda=0.1,
    rng: Optional[np.random.Generator] = None,
) -> SpatialDifferentialExpressionResult:
    if rng is None:
        rng = np.random.default_rng()
    r_flat = deconvolution_result.reads_trace.mean(
        axis=0
    )  # <N Spots> x <N Genes> x <N Cell Types>
    r_flat = r_flat.transpose(2, 1, 0)  # <N Cell Types> x <N Genes> x <N Spots>
    r_flat = np.clip(r_flat, 1e-10, np.inf)
    trendfilter_indices = edges_to_linear_tf(data.edges)
    n_spots = data.n_spot_in
    n_genes = data.n_gene

    t_log_rates = torch.tensor(np.log(r_flat))
    t_Reads = torch.tensor(data.counts.T)
    model = SpatialPrograms(
        log_rates=t_log_rates,
        n_cell_types=deconvolution_result.n_components,
        n_programs=n_programs,
        n_genes=n_genes,
        n_spots=n_spots,
        rng=rng,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    losses = []

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
        loss = (
            -t_log_like
            + trend_filtering_lambda * t_trend_filter
            + lasso_lambda * t_lasso
        )

        # Calculate gradients
        loss.backward()

        # Apply the update
        optimizer.step()

        losses.append(loss.item())

    model.eval()
    w_hat = model.W.detach().numpy()  # (n_cell_types, n_programs, n_spots)
    v_hat = model.V.detach().numpy()  # (n_cell_types, n_genes, n_programs)

    return SpatialDifferentialExpressionResult(
        w_hat=w_hat, v_hat=v_hat, losses=np.array(losses)
    )


def plot_loss(stp: SpatialDifferentialExpressionResult, output_path: str):
    fig, ax = plt.subplots()
    ax.set_title("STP Training Loss")
    ax.plot(stp.losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    fig.savefig(output_path)
    plt.close(fig)


def plot_spatial_transcriptional_programs(
    stp: SpatialDifferentialExpressionResult,
    data: SpatialExpressionDataset,
    output_dir: str,
    output_format: str = "pdf",
):
    for cell_type_idx in range(stp.n_components):
        n_panels_x = min(3, stp.n_spatial_patterns)
        n_panels_y = np.ceil(stp.n_spatial_patterns / n_panels_x).astype(int)

        fig = plt.figure(
            figsize=(
                n_panels_x * 5,
                n_panels_y * 5,
            )
        )
        gs = gridspec.GridSpec(
            nrows=n_panels_y, ncols=n_panels_x, wspace=0.22, hspace=0.3
        )

        fig.suptitle(f"Cell Type {cell_type_idx} Spatial Programs")

        for program_idx in range(stp.n_spatial_patterns):
            ax = fig.add_subplot(gs[program_idx])

            ax.set_title(f"Program {program_idx}")
            values = stp.w_hat[cell_type_idx, program_idx, :]
            ax, cb, norm, hcoord, vcoord = plot_colored_spatial_polygon(
                fig=fig,
                ax=ax,
                coords=data.positions_tissue,
                values=values,
                layout=data.layout,
                colormap=cm.coolwarm,
                norm=TwoSlopeNorm(
                    vmin=min(values.min(), -1e-6),
                    vcenter=0,
                    vmax=max(values.max(), 1e-6),
                ),
            )
            cb.ax.set_yscale("linear")
            ax.set_axis_off()
        fig.savefig(
            os.path.join(output_dir, f"cell_type_{cell_type_idx}_stp.{output_format}")
        )
        plt.close(fig)


def plot_top_spatial_program_genes(
    stp: SpatialDifferentialExpressionResult,
    data: SpatialExpressionDataset,
    output_dir: str,
    output_format: str = "pdf",
    n_top_genes: int = 5,
):
    for cell_type_idx in range(stp.n_components):
        for program_idx in range(stp.n_spatial_patterns):
            n_panels_x = min(3, n_top_genes)
            n_panels_y = np.ceil(n_top_genes / n_panels_x).astype(int)

            fig = plt.figure(
                figsize=(
                    n_panels_x * 5,
                    n_panels_y * 5,
                )
            )
            gs = gridspec.GridSpec(
                nrows=n_panels_y, ncols=n_panels_x, wspace=0.22, hspace=0.3
            )

            values = stp.v_hat[cell_type_idx, :, program_idx]
            sorted_order = np.argsort(np.abs(values))[::-1]
            gene_names = data.gene_names[sorted_order[:n_top_genes]]
            gene_indices = sorted_order[:n_top_genes]

            fig.suptitle(f"Cell Type {cell_type_idx} Program {program_idx} Top Genes")

            for plot_idx, (gene_idx, gene_name) in enumerate(
                zip(gene_indices, gene_names)
            ):
                ax = fig.add_subplot(gs[plot_idx])

                plot_colored_spatial_polygon(
                    fig=fig,
                    ax=ax,
                    coords=data.positions_tissue,
                    values=data.counts[:, gene_idx],
                    layout=data.layout,
                    colormap=cm.coolwarm,
                )

                ax.set_title(f"{gene_name}")
                ax.set_axis_off()
            fig.savefig(
                os.path.join(
                    output_dir,
                    f"cell_type_{cell_type_idx}_program_{program_idx}_top_{n_top_genes}_genes.{output_format}",
                )
            )
            plt.close(fig)
