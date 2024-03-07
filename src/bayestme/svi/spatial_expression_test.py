from bayestme.svi.spatial_expression import model
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
from pyro.infer import SVI, Trace_ELBO
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

    result = bayestme.svi.deconvolution.deconvolve(
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

    reads = torch.tensor(result.reads_trace.mean(0))

    args = {"reads": reads, "y": torch.tensor(stdata.counts), "k": 2, "h": 2}

    optimizer = Adam(optim_args={"lr": 0.05})
    guide = AutoNormal(poutine.block(model, hide=["y_h"]))

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    for step in tqdm.trange(4000):  # Consider running for more steps.
        loss = svi.step(**args)

    params = pyro.get_param_store()

    result = defaultdict(list)
    for _ in tqdm.trange(200):
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

    samples = {k: np.stack(v).mean(axis=0) for k, v in result.items()}

    print(samples)
