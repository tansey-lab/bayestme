import os.path

from bayestme.svi.spatial_expression import model, get_loss_for_seed, config_enumerate
import pyro
import torch
import numpy as np
from pyro import poutine
import numpy as np
import pyro
import pyro.util
import torch
import tqdm
from pyro import poutine
from pyro.distributions import Gamma, Dirichlet
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from torch.distributions import biject_to
from bayestme.synthetic_data import generate_demo_stp_dataset
import bayestme.svi.deconvolution
from collections import defaultdict
import pyro.distributions as dist
from matplotlib import pyplot as plt
from bayestme.plot.common import plot_gene_in_tissue_counts
from bayestme.plot.deconvolution import plot_deconvolution
from bayestme.data import (
    add_deconvolution_results_to_dataset,
    SpatialExpressionDataset,
    DeconvolutionResult,
)
from bayestme.utils import construct_trendfilter
from bayestme.utils import construct_edge_adjacency
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete


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

    K = 2
    n_traces = 100
    svi_steps = 10_000

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

    r_igk = torch.tensor(
        np.transpose(
            (deconv_result.cell_num_trace.mean(0) * deconv_result.beta_trace.mean(0).T)[
                :, :, None
            ]
            * deconv_result.expression_trace.mean(0)[None, :, :],
            (0, 2, 1),
        )
    )

    pyro.clear_param_store()
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

        # best_loss, best_seed = min(
        #    [get_loss_for_seed(seed, optimizer, elbo, args) for seed in range(1000)]
        # )
        # print(best_loss, best_seed)
        # pyro.set_rng_seed(best_seed)
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

        result = defaultdict(list)
        for _ in tqdm.trange(500):
            guide_trace = poutine.trace(guide).get_trace(**args)  # record the globals
            trained_model = poutine.replay(model, trace=guide_trace)

            def classifier(args, temperature=0):
                inferred_model = infer_discrete(
                    trained_model, temperature=temperature, first_available_dim=-3
                )  # avoid conflict with data plate
                trace = poutine.trace(inferred_model).get_trace(**args)
                return trace.nodes["h"]["value"]

            h_for_real = classifier(args)

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
            for name, v in sample.items():
                result[name].append(v)

        all_h_values = np.stack(result["h"])

        h_modes = np.zeros_like(all_h_values[0, ...])
        h_freqs = np.zeros_like(all_h_values[0, ...]).astype(float)

        for g in range(all_h_values.shape[1]):
            vals, counts = np.unique(all_h_values[:, g], return_counts=True)
            h_modes[g] = vals[counts.argmax()]
            h_freqs[g] = float(counts.max()) / float(counts.sum())

        samples = {name: np.stack(v).mean(axis=0) for name, v in result.items()}

        for h in range(5):
            if h == 0:
                continue
            img = np.zeros((20, 20))
            weights = samples["w"][h, :]

            img[stdata.positions[:, 0], stdata.positions[:, 1]] = weights

            plt.imshow(img)
            plt.savefig(f"w_k_{k}_h_{h}.png")

        print(samples, h_modes, h_freqs)


if __name__ == "__main__":
    test_model_pipeline()
