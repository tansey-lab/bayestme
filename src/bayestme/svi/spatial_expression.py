import pyro
import pyro.distributions as dist
import torch
from pyro.infer.enum import config_enumerate
from pyro.infer.autoguide import AutoNormal
from pyro.infer.svi import SVI
from pyro import poutine


def init_loc_fn(site, y_igk=None, h=None, **kwargs):
    i = y_igk.shape[0]
    g = y_igk.shape[1]
    k = y_igk.shape[2]

    if site["name"] == "v":
        return torch.distributions.Normal(0, 1).sample((k, g, 1))
    elif site["name"] == "p":
        return torch.distributions.Dirichlet(torch.ones(h)).sample((k, g))[
            :, :, None, :
        ]
    elif site["name"] == "c":
        return torch.distributions.Normal(0, 1).sample((k, g, 1))
    elif site["name"] == "h":
        return torch.randint(0, h, (1, g, k))
    elif site["name"] == "w":
        return torch.distributions.Normal(0, 1).sample((h, k, 1, i))
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
def model(r_igk, y_igk, h=None, alpha0_hparam=10, alpha_hparam=1):
    """
    Model for spatial expression
    """
    i = y_igk.shape[0]
    g = y_igk.shape[1]
    k = y_igk.shape[2]

    spot_plate = pyro.plate("spot", i, dim=-1)
    gene_plate = pyro.plate("gene", g, dim=-2)
    component_plate = pyro.plate("component", k, dim=-3)
    stp_plate = pyro.plate("stp", h, dim=-4)

    alpha = torch.ones(h) * alpha_hparam
    alpha[0] = alpha0_hparam

    with spot_plate, component_plate, stp_plate:
        w = pyro.sample("w", dist.Normal(0, 1))

    w[..., 0, :, :, :] *= 0.0
    w = w[..., torch.zeros(g, dtype=torch.long), :]

    with gene_plate, component_plate:
        v = pyro.sample("v", dist.Normal(0, 1))
        p = pyro.sample("p", dist.Dirichlet(alpha))
        c = pyro.sample("c", dist.Normal(0, 1))
        h = pyro.sample("h", dist.Categorical(p))

        h = h[..., torch.zeros(i, dtype=torch.long)]

        with spot_plate:
            w_h = w[
                ...,
                h,
                torch.arange(h.shape[-3])[:, None, None],
                torch.arange(h.shape[-2])[None, :, None],
                torch.arange(h.shape[-1])[None, None, :],
            ]

            theta = pyro.deterministic("theta", torch.sigmoid(w_h * v + c))
            pyro.sample("y_h", dist.NegativeBinomial(r_igk.T, theta), obs=y_igk.T.int())
