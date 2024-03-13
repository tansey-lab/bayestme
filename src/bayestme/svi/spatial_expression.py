import pyro
import pyro.distributions as dist
import torch
from pyro.infer.enum import config_enumerate
from pyro.infer.autoguide import AutoNormal
from pyro.infer.svi import SVI
from pyro import poutine
from pyro.ops.indexing import Vindex


def init_loc_fn(site, y_ig=None, h=None, **kwargs):
    i = y_ig.shape[0]
    g = y_ig.shape[1]

    if site["name"] == "v":
        return torch.distributions.Normal(0, 1).sample((g, 1))
    elif site["name"] == "p":
        return torch.distributions.Dirichlet(torch.ones(h)).sample()
    elif site["name"] == "c":
        return torch.distributions.Normal(0, 1).sample((g, 1))
    elif site["name"] == "h":
        return torch.randint(0, h, (g, 1))
    elif site["name"] == "w":
        return torch.distributions.Normal(0, 1).sample((h, 1, i))
    else:
        raise ValueError(site["name"])


def get_loss_for_seed(seed, optim, elbo, kwargs):
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    guide = AutoNormal(
        poutine.block(model, hide=["h"]),
        init_loc_fn=lambda site: init_loc_fn(site=site, **kwargs),
    )
    svi = SVI(model, guide, optim, loss=elbo)
    return svi.loss(model, guide, **kwargs), seed


@config_enumerate(default="parallel")
def model(r_ig, y_ig, h=None, alpha0_hparam=10, alpha_hparam=1):
    """
    Model for spatial expression
    """
    i = y_ig.shape[0]
    g = y_ig.shape[1]

    spot_plate = pyro.plate("spot", i, dim=-1)
    gene_plate = pyro.plate("gene", g, dim=-2)
    stp_plate = pyro.plate("stp", h, dim=-3)

    alpha = torch.ones(h) * alpha_hparam
    alpha[0] = alpha0_hparam

    with spot_plate, stp_plate:
        w = pyro.sample("w", dist.Normal(0, 1))

    w[0] *= 0.0
    p = pyro.sample("p", dist.Dirichlet(alpha).to_event())

    with gene_plate:
        v = pyro.sample("v", dist.Normal(0, 1))
        c = pyro.sample("c", dist.Normal(0, 1))
        h = pyro.sample("h", dist.Categorical(p))

        with spot_plate:
            w_h = Vindex(w)[h, :]

            theta = pyro.deterministic("theta", torch.sigmoid(w_h * v + c))
            pyro.sample("y_h", dist.NegativeBinomial(r_ig.T, theta), obs=y_ig.T.int())
