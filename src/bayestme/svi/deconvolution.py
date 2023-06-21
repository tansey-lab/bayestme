import logging
from collections import defaultdict
from typing import Optional

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
    ):
        # Obs:  ST count mat
        #       etiher np array or torch tensor
        # pos:  spot locations
        #       np array
        # lr:   learning rate
        # beta: Adam decay params
        self.opt_params = {"lr": lr, "betas": (beta_1, beta_2)}
        self.optimizer = Adam(self.opt_params)

        self.Obs = torch.tensor(stdata.counts)
        self.N = stdata.n_spot_in
        self.G = stdata.n_gene
        self.edges = get_edges(stdata.positions_tissue, layout=stdata.layout.value)
        self.D = construct_edge_adjacency(self.edges)
        self.D = construct_trendfilter(self.D, 0)
        self.sp_reg_coeff = rho

    def model(self, data, n_class, n_genes):
        # expression coeff
        a_0 = torch.tensor(100.0)
        b_0 = torch.tensor(1.0)
        beta = pyro.sample("exp_load", dist.Gamma(a_0, b_0).expand([self.K]).to_event())
        # expression profile
        alpha_0 = torch.ones(self.G)
        phi = pyro.sample(
            "exp_profile", dist.Dirichlet(alpha_0).expand([self.K]).to_event()
        )
        # expression
        celltype_exp = beta[:, None] * phi

        # cell type probs
        psi_0 = torch.ones(self.K)
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
            "beta_a", torch.ones(self.K) * 100.0, constraint=constraints.positive
        )
        beta_b = pyro.param(
            "beta_b", torch.tensor(1.0), constraint=constraints.positive
        )
        beta = pyro.sample("exp_load", dist.Gamma(beta_a, beta_b).to_event())

        phi_a = pyro.param(
            "phi_a", torch.ones(self.K, self.G), constraint=constraints.positive
        )
        phi = pyro.sample("exp_profile", dist.Dirichlet(phi_a).to_event())

        psi_a = pyro.param(
            "psi_a", torch.ones(self.N, self.K), constraint=constraints.positive
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
            "beta_a", torch.ones(self.K) * 100.0, constraint=constraints.positive
        )
        beta_b = pyro.param(
            "beta_b", torch.tensor(1.0), constraint=constraints.positive
        )
        beta = pyro.sample("exp_load", dist.Gamma(beta_a, beta_b).to_event())

        phi_a = pyro.param(
            "phi_a", torch.ones(self.K, self.G), constraint=constraints.positive
        )
        phi = pyro.sample("exp_profile", dist.Dirichlet(phi_a).to_event())

        psi_a = pyro.param(
            "psi_a", torch.ones(self.N, self.K), constraint=constraints.positive
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
        return torch.abs(self.Delta @ x.reshape(-1, 1)).sum() * self.sp_reg_coeff

    def deconvolution(self, K, n_iter=10000, n_traces=1000, use_spatial_guide=True):
        self.K = K
        # TODO: maybe make Delta sparse, but need to change spatial_regularizer as well (double check if sparse grad is supported)
        self.Delta = torch.kron(torch.eye(self.K), self.D.to_dense())
        logger.info("Optimizer: {} {}".format("Adam", self.opt_params))
        logger.info(
            "Deconvolving: {} spots, {} genes, {} cell types".format(
                self.N, self.G, self.K
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
            svi.step(self.Obs, self.K, self.G)

        result = defaultdict(list)
        for _ in tqdm.trange(n_traces):
            guide_trace = poutine.trace(guide).get_trace(self.Obs, self.K, self.G)
            model_trace = poutine.trace(
                poutine.replay(self.model, guide_trace)
            ).get_trace(self.Obs, self.K, self.G)
            sample = {
                name: site["value"]
                for name, site in model_trace.nodes.items()
                if (
                    (site["type"] == "sample")
                    and (
                        (not site.get("is_observed", True))
                        or (site.get("infer", False).get("_deterministic", False))
                    )
                    and not isinstance(
                        site.get("fn", None), poutine.subsample_messenger._Subsample
                    )
                )
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
            lam2=self.sp_reg_coeff,
            n_components=K,
        )


def deconvolve(
    stdata: SpatialExpressionDataset,
    n_components=None,
    rho=None,
    n_svi_steps=10_000,
    n_samples=100,
    use_spatial_guide=True,
    rng: Optional[np.random.Generator] = None,
) -> data.DeconvolutionResult:
    if rng:
        try:
            seed_sequence = np.random.SeedSequence(rng.__getstate__()["state"]["state"])
            states = seed_sequence.generate_state(3)
            pyro.util.set_rng_state(
                {
                    "pyro": states[0],
                    "torch": states[1],
                    "numpy": states[2],
                }
            )
        except KeyError:
            logger.warning("RNG state init failed, using default")

    return BayesTME_VI(
        stdata=stdata,
        rho=rho,
    ).deconvolution(
        n_traces=n_samples,
        n_iter=n_svi_steps,
        K=n_components,
        use_spatial_guide=use_spatial_guide,
    )
