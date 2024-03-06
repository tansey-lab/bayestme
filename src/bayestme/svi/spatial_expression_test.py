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
    stdata = generate_demo_stp_dataset()

    K = 2
    n_traces = 100

    rng = np.random.default_rng(42)

    result = bayestme.svi.deconvolution.deconvolve(
        stdata=stdata,
        n_components=K,
        rho=0.5,
        n_svi_steps=1000,
        n_samples=n_traces,
        use_spatial_guide=True,
        expression_truth=None,
        rng=rng,
    )
