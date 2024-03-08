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


def test_model():
    pyro.util.set_rng_seed(0)
    expected_exp = torch.tensor(np.random.random((64, 5)))

    trace = poutine.trace(model).get_trace(
        expected_exp,
        y=None,
        k=3,
        h=10,
    )
    trace.compute_log_prob()
    print("---------- Tensor Shapes ------------")
    print(trace.format_shapes())

    optimizer = Adam(optim_args={"lr": 0.05})
    guide = AutoNormal(poutine.block(model, hide=["y"]))

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    for step in tqdm.trange(1):  # Consider running for more steps.
        loss = svi.step(expected_exp, y=None, k=3, h=10)


def test_model_pipeline():
    pyro.util.set_rng_seed(0)
    stdata = generate_demo_stp_dataset()

    K = 2
    n_traces = 100

    rng = np.random.default_rng(42)

    deconv_result = bayestme.svi.deconvolution.deconvolve(
        stdata=stdata,
        n_components=K,
        rho=0.5,
        n_svi_steps=10_000,
        n_samples=n_traces,
        use_spatial_guide=False,
        expression_truth=None,
        rng=rng,
    )

    pyro.clear_param_store()
    h = 4

    y_igk = torch.tensor(deconv_result.reads_trace.mean(axis=0))
    args = {"y_igk": y_igk, "h": h, "alpha0_hparam": 1.0}

    optimizer = Adam(optim_args={"lr": 0.05})
    guide = AutoNormal(poutine.block(model, hide=["h"]))

    elbo = TraceEnum_ELBO()

    best_loss, best_seed = min(
        [get_loss_for_seed(seed, optimizer, elbo, y_igk, h) for seed in range(1000)]
    )
    print(best_loss, best_seed)
    pyro.set_rng_seed(best_seed)
    pyro.clear_param_store()

    svi = SVI(model, guide, optimizer, loss=elbo)

    for step in tqdm.trange(10_000):  # Consider running for more steps.
        loss = svi.step(**args)

    params = pyro.get_param_store()

    result = defaultdict(list)
    for _ in tqdm.trange(500):
        guide_trace = poutine.trace(guide).get_trace(**args)
        model_trace = poutine.trace(poutine.replay(model, guide_trace)).get_trace(
            **args
        )
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

    all_h_values = np.stack(result["h"])[:, :, :, 0]

    h_modes = np.zeros_like(all_h_values[0, ...])
    h_freqs = np.zeros_like(all_h_values[0, ...]).astype(float)

    for k in range(all_h_values.shape[1]):
        for g in range(all_h_values.shape[2]):
            vals, counts = np.unique(all_h_values[:, k, g], return_counts=True)
            h_modes[k, g] = vals[counts.argmax()]
            h_freqs[k, g] = float(counts.max()) / float(counts.sum())

    samples = {k: np.stack(v).mean(axis=0) for k, v in result.items()}

    print(samples, h_modes, h_freqs)
