import os.path
from collections import defaultdict

import h5py
import numpy as np
import pyro
import pyro.util
import torch
import tqdm
from pyro import poutine
from pyro.infer import TraceEnum_ELBO, infer_discrete
from pyro.infer.autoguide import AutoNormal

import bayestme.svi.deconvolution
from bayestme.data import (
    add_deconvolution_results_to_dataset,
    SpatialExpressionDataset,
    DeconvolutionResult,
)
from bayestme.plot.common import plot_gene_in_tissue_counts
from bayestme.svi.spatial_expression import model
from bayestme.synthetic_data import generate_demo_stp_dataset
from bayestme.utils import construct_edge_adjacency
from bayestme.utils import construct_trendfilter


def spatial_loss(D):
    w = pyro.param("AutoNormal.locs.w")
    w = w[1:]
    # h x i tensor
    penalty = torch.abs(D @ w.T).sum()
    return penalty


def test_model_pipeline():
    pyro.util.set_rng_seed(0)
    stdata = generate_demo_stp_dataset(width=20, height=20)

    adjacency_matrix = construct_edge_adjacency(stdata.edges)

    D = torch.tensor(
        construct_trendfilter(adjacency_matrix, k=1, sparse=False).todense()
    ).float()

    plot_gene_in_tissue_counts(
        stdata, gene="north_marker", output_file="north_marker_gene_expression.png"
    )
    plot_gene_in_tissue_counts(
        stdata, gene="south_marker", output_file="south_marker_gene_expression.png"
    )
    plot_gene_in_tissue_counts(
        stdata, gene="north_stp", output_file="stp_gene_expression.png"
    )
    plot_gene_in_tissue_counts(
        stdata, gene="north_stp2", output_file="stp2_gene_expression.png"
    )

    K = 2
    n_traces = 100
    svi_steps = 100_000

    rng = np.random.default_rng(42)

    if not os.path.exists("./decon_results.h5"):
        deconv_result = bayestme.svi.deconvolution.deconvolve(
            stdata=stdata,
            n_components=K,
            rho=0.5,
            n_svi_steps=svi_steps,
            n_samples=n_traces,
            use_spatial_guide=False,
            expression_truth=None,
            rng=rng,
        )

        add_deconvolution_results_to_dataset(stdata=stdata, result=deconv_result)
        stdata.save("./stdata.h5")
        deconv_result.save("./decon_results.h5")
    else:
        stdata = SpatialExpressionDataset.read_h5("./stdata.h5")
        deconv_result = DeconvolutionResult.read_h5("./decon_results.h5")

    pyro.clear_param_store()

    r_igk = torch.tensor(
        np.transpose(
            (deconv_result.cell_num_trace.mean(0) * deconv_result.beta_trace.mean(0).T)[
                :, :, None
            ]
            * deconv_result.expression_trace.mean(0)[None, :, :],
            (0, 2, 1),
        )
    )

    h = 2

    y_igk = torch.tensor(deconv_result.reads_trace.mean(axis=0))

    for k in range(K):
        args = {
            "r_ig": r_igk[:, :, k],
            "y_ig": y_igk[:, :, k],
            "h": h,
            "alpha0_hparam": 10.0,
        }

        guide = AutoNormal(poutine.block(model, hide=["h"]))

        elbo = TraceEnum_ELBO(max_plate_nesting=2)

        pyro.clear_param_store()

        loss_fn = lambda model, guide: elbo.differentiable_loss(model, guide, **args)
        with pyro.poutine.trace(param_only=True) as param_capture:
            loss = loss_fn(model, guide)
        params = set(
            site["value"].unconstrained() for site in param_capture.trace.nodes.values()
        )
        optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.90, 0.999))
        for i in tqdm.trange(svi_steps):
            # compute loss
            loss = loss_fn(model, guide)
            spatial_loss_value = spatial_loss(D)

            loss += spatial_loss_value
            loss.backward()
            # take a step and zero the parameter gradients
            optimizer.step()
            optimizer.zero_grad()

        results = defaultdict(list)

        for _ in tqdm.trange(100):
            guide_trace = poutine.trace(guide).get_trace(**args)  # record the globals
            trained_model = poutine.replay(model, trace=guide_trace)

            def classifier(args, temperature=0):
                inferred_model = infer_discrete(
                    trained_model, temperature=temperature, first_available_dim=-3
                )  # avoid conflict with data plate
                trace = poutine.trace(inferred_model).get_trace(**args)
                return trace

            trace = classifier(args)

            for name, v in trace.nodes.items():
                if name in ["v", "p", "c", "h", "w"]:
                    results[name].append(v["value"].detach().numpy())

        with h5py.File(f"stp_k_{k}.h5", "w") as f:
            for name, v in results.items():
                f.create_dataset(name=name, data=v, compression="gzip")


if __name__ == "__main__":
    test_model_pipeline()
