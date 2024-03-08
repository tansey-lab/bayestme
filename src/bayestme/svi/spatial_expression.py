import pyro
import pyro.distributions as dist
import torch
from pyro.infer.enum import config_enumerate


@config_enumerate(default="parallel")
def model(y_igk, h=None, alpha0_hparam=10, alpha_hparam=1):
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

    y = y_igk.sum(dim=-1).int()

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
            pyro.sample("y_h", dist.NegativeBinomial(y.T, theta), obs=y_igk.T.int())
