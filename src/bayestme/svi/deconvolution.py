import logging
from collections import defaultdict
from typing import Optional

import random
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
import tqdm
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from bayestme import data
from bayestme.data import SpatialExpressionDataset, DeconvolutionResult
from bayestme.utils import get_edges
from bayestme.common import ArrayType
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def construct_edge_adjacency(neighbors):
    data, rows, cols = [], [], []
    nrows = 0
    for i, j in neighbors:
        data.extend([1, -1])
        rows.extend([nrows, nrows])
        cols.extend([i, j])
        nrows += 1
    indices = torch.tensor([rows, cols])
    values = torch.tensor(data)
    edge_adjacency_matrix = torch.sparse_coo_tensor(indices, values).float()
    return edge_adjacency_matrix


def construct_trendfilter(adjacency_matrix, k):
    transformed_edge_adjacency_matrix = adjacency_matrix.clone()
    for i in range(k):
        if i % 2 == 0:
            transformed_edge_adjacency_matrix = adjacency_matrix.t().mm(
                transformed_edge_adjacency_matrix
            )
        else:
            transformed_edge_adjacency_matrix = adjacency_matrix.mm(
                transformed_edge_adjacency_matrix
            )
    extra = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]),
        torch.tensor([1]),
        size=(1, transformed_edge_adjacency_matrix.shape[1]),
    )
    transformed_edge_adjacency_matrix = torch.vstack(
        [transformed_edge_adjacency_matrix, extra]
    )
    return transformed_edge_adjacency_matrix


def rv_should_be_sampled(site):
    """
    Determine if a random variable in the model should be saved or
    if its redundant.
    """
    return (
        (site["type"] == "sample")
        and (
            (not site.get("is_observed", True))
            or (site.get("infer", False).get("_deterministic", False))
        )
        and not isinstance(site.get("fn", None), poutine.subsample_messenger._Subsample)
    )


def create_reads_trace(psi, exp_profile, exp_load, cell_num_total):
    """
    :param psi: <N tissue spots> x <N components> matrix
    :param exp_profile: <N components> x <N markers> matrix
    :param exp_load: <N components> matrix
    :param cell_num_total: <N tissue spots> matrix
    :return: <N tissue spots> x <N markers> x <N components> matrix
    """
    number_of_cells_per_component = (psi.T * cell_num_total).T * exp_load.T
    result = number_of_cells_per_component[:, :, None] * exp_profile[None, :, :]
    return np.transpose(result, (0, 2, 1))


class BayesTME_VI:
    def __init__(
        self,
        stdata: SpatialExpressionDataset,
        rho=0.5,
        lr=0.001,
        beta_1=0.90,
        beta_2=0.999,
        expression_truth: Optional[ArrayType] = None,
        expression_truth_weight=10.0,
        expression_truth_n_dummy_cell_types=2,
    ):
        # Obs:  ST count mat
        #       etiher np array or torch tensor
        # pos:  spot locations
        #       np array
        # lr:   learning rate
        # beta: Adam decay params
        self.opt_params = {"lr": lr, "betas": (beta_1, beta_2)}
        self.optimizer = Adam(self.opt_params)

        self.counts = torch.tensor(stdata.counts)
        self.N = stdata.n_spot_in
        self.n_genes = stdata.n_gene
        self.edges = get_edges(stdata.positions_tissue, layout=stdata.layout)
        self.delta = construct_edge_adjacency(self.edges)
        self.delta = construct_trendfilter(self.delta, 0)
        self.spatial_regularization_coefficient = rho
        self.losses = []
        self.expression_truth_weight = expression_truth_weight
        self.expression_truth_n_dummy_cell_types = expression_truth_n_dummy_cell_types
        if expression_truth is not None:
            self.expression_truth = expression_truth
        else:
            self.expression_truth = None

    def model(self, data, n_class, n_genes):
        # expression coeff
        a_0 = torch.tensor(100.0)
        b_0 = torch.tensor(1.0)
        beta = pyro.sample(
            "exp_load", dist.Gamma(a_0, b_0).expand([self.n_celltypes]).to_event()
        )
        # expression profile
        alpha_0 = torch.ones(self.n_genes)
        if self.expression_truth is None:
            phi = pyro.sample(
                "exp_profile",
                dist.Dirichlet(alpha_0).expand([self.n_celltypes]).to_event(),
            )
        else:
            exp_truth_weighted = (
                self.expression_truth * self.expression_truth_weight * self.n_genes
            )

            for _ in range(self.expression_truth_n_dummy_cell_types):
                exp_truth_weighted = np.concatenate(
                    [exp_truth_weighted, np.ones(self.n_genes)[None, :]]
                )

            phi = pyro.sample(
                "exp_profile",
                dist.Dirichlet(torch.tensor(exp_truth_weighted)).to_event(1),
            )
        # expression
        celltype_exp = beta[:, None] * phi

        # cell type probs
        psi_0 = torch.ones(self.n_celltypes)
        psi = pyro.sample("psi", dist.Dirichlet(psi_0).expand([self.N]).to_event())
        # cell numbers
        d_a = torch.tensor(10.0)
        d_b = torch.tensor(1.0)
        cell_num = pyro.sample(
            "cell_num_total", dist.Gamma(d_a, d_b).expand([self.N]).to_event()
        )
        # TODO: maybe make this pyro.deterministic or Normal(cell_num[:, None] * psi, sigma) or something
        d = cell_num[:, None] * psi

        # exprected expression
        expected_exp = d @ celltype_exp
        # TODO: potentially can speed this up further
        #       calc something like log_prob = (data_flat * log(expected_exp_flat) - expected_exp_flat).sum()
        #       do pyro.factor('obs', log_prob)
        return pyro.sample("obs", dist.Poisson(expected_exp).to_event(), obs=data)

    def guide(self, data, n_class, n_genes):
        """
        guide without spatial regularizer
        """
        beta_a = pyro.param(
            "beta_a",
            torch.ones(self.n_celltypes) * 100.0,
            constraint=constraints.positive,
        )
        beta_b = pyro.param(
            "beta_b", torch.tensor(1.0), constraint=constraints.positive
        )
        beta = pyro.sample("exp_load", dist.Gamma(beta_a, beta_b).to_event())

        phi_a = pyro.param(
            "phi_a",
            torch.ones(self.n_celltypes, self.n_genes),
            constraint=constraints.positive,
        )
        phi = pyro.sample("exp_profile", dist.Dirichlet(phi_a).to_event())

        psi_a = pyro.param(
            "psi_a",
            torch.ones(self.N, self.n_celltypes),
            constraint=constraints.positive,
        )
        psi = pyro.sample("psi", dist.Dirichlet(psi_a).to_event())

        d_a = pyro.param(
            "d_a", torch.ones(self.N) * 20, constraint=constraints.positive
        )
        d_b = pyro.param("d_b", torch.tensor(1.0), constraint=constraints.positive)
        cell_num = pyro.sample("cell_num_total", dist.Gamma(d_a, d_b).to_event())

    def spatial_guide(self, data, n_class, n_genes):
        """
        guide with spatial regularizer
        """
        beta_a = pyro.param(
            "beta_a",
            torch.ones(self.n_celltypes) * 100.0,
            constraint=constraints.positive,
        )
        beta_b = pyro.param(
            "beta_b", torch.tensor(1.0), constraint=constraints.positive
        )
        beta = pyro.sample("exp_load", dist.Gamma(beta_a, beta_b).to_event())

        phi_a = pyro.param(
            "phi_a",
            torch.ones(self.n_celltypes, self.n_genes),
            constraint=constraints.positive,
        )
        phi = pyro.sample("exp_profile", dist.Dirichlet(phi_a).to_event())

        psi_a = pyro.param(
            "psi_a",
            torch.ones(self.N, self.n_celltypes),
            constraint=constraints.positive,
        )
        psi = pyro.sample("psi", dist.Dirichlet(psi_a).to_event())
        # spatial regularizer
        pyro.factor("regularizer", self.spatial_regularizer(psi), has_rsample=True)

        d_a = pyro.param(
            "d_a", torch.ones(self.N) * 20, constraint=constraints.positive
        )
        d_b = pyro.param("d_b", torch.tensor(1.0), constraint=constraints.positive)
        cell_num = pyro.sample("cell_num_total", dist.Gamma(d_a, d_b).to_event())

    def spatial_regularizer(self, x):
        # Delta should be of size (n_edges * n_celltype) by (n_spot * n_celltype)
        # x should be of size n_spot by n_celltype
        return (
            torch.abs(self.Delta @ x.reshape(-1, 1)).sum()
            * self.spatial_regularization_coefficient
        )

    def deconvolution(self, K, n_iter=10000, n_traces=1000, use_spatial_guide=True):
        if self.expression_truth is not None:
            self.n_celltypes = (
                self.expression_truth.shape[0]
                + self.expression_truth_n_dummy_cell_types
            )
        else:
            self.n_celltypes = K
        # TODO: maybe make Delta sparse, but need to change spatial_regularizer as well (double check if sparse grad is supported)
        self.Delta = torch.kron(torch.eye(self.n_celltypes), self.delta.to_dense())
        logger.info("Optimizer: {} {}".format("Adam", self.opt_params))
        logger.info(
            "Deconvolving: {} spots, {} genes, {} cell types".format(
                self.N, self.n_genes, self.n_celltypes
            )
        )

        if use_spatial_guide:
            guide = self.spatial_guide
            logger.info("with spatial regularizer")
        else:
            guide = self.guide
            logger.info("without spatial regularizer")

        pyro.clear_param_store()
        svi = SVI(self.model, guide, self.optimizer, loss=Trace_ELBO())
        for step in tqdm.trange(n_iter):
            self.losses.append(svi.step(self.counts, self.n_celltypes, self.n_genes))

        result = defaultdict(list)
        for _ in tqdm.trange(n_traces):
            guide_trace = poutine.trace(guide).get_trace(
                self.counts, self.n_celltypes, self.n_genes
            )
            model_trace = poutine.trace(
                poutine.replay(self.model, guide_trace)
            ).get_trace(self.counts, self.n_celltypes, self.n_genes)
            sample = {
                name: site["value"]
                for name, site in model_trace.nodes.items()
                if rv_should_be_sampled(site)
            }
            sample = {name: site.detach().numpy() for name, site in sample.items()}
            for k, v in sample.items():
                result[k].append(v)

            result["read_trace"].append(
                create_reads_trace(
                    psi=sample["psi"],
                    exp_profile=sample["exp_profile"],
                    exp_load=sample["exp_load"],
                    cell_num_total=sample["cell_num_total"],
                )
            )

        samples = {k: np.stack(v) for k, v in result.items()}
        return DeconvolutionResult(
            cell_prob_trace=samples["psi"],
            expression_trace=samples["exp_profile"],
            beta_trace=samples["exp_load"],
            cell_num_trace=(samples["cell_num_total"].T * samples["psi"].T).T,
            reads_trace=samples["read_trace"],
            lam2=self.spatial_regularization_coefficient,
            n_components=self.n_celltypes,
            losses=np.array(self.losses),
        )

    def plot_loss(self, output_file):
        """
        Plot the loss curve

        :param output_file: Where to save the plot
        """
        fig, ax = plt.subplots(1)
        ax.plot(np.arange(len(self.losses)), self.losses)
        ax.set_xlabel("Step Number")
        ax.set_ylabel("Loss")
        fig.savefig(output_file)


def deconvolve(
    stdata: SpatialExpressionDataset,
    n_components=None,
    rho=None,
    n_svi_steps=10_000,
    n_samples=100,
    use_spatial_guide=True,
    expression_truth=None,
    rng: Optional[np.random.Generator] = None,
) -> data.DeconvolutionResult:
    if rng:
        try:
            seed_sequence = np.random.SeedSequence(rng.__getstate__()["state"]["state"])
            states = seed_sequence.generate_state(3)
            np.random.seed(states[0])
            torch.manual_seed(int(states[1]))
            random.seed(int(states[2]))
        except KeyError:
            logger.warning("RNG state init failed, using default")

    svi = BayesTME_VI(
        stdata=stdata,
        rho=rho,
        expression_truth=expression_truth,
    )
    return svi.deconvolution(
        n_traces=n_samples,
        n_iter=n_svi_steps,
        K=n_components,
        use_spatial_guide=use_spatial_guide,
    )
