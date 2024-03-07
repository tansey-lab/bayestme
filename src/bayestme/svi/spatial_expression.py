import pyro
import pyro.distributions as dist
import torch


def model(data=None, h=None, alpha0_hparam=10, alpha_hparam=1):
    """
    Model for spatial expression
    :param expected_exp: Expected expression N spot x G genes

    """
    i = data.shape[0]
    g = data.shape[1]
    k = data.shape[2]

    w = pyro.sample("w", dist.Normal(0, 1).expand([i, k, h]).to_event(3))

    v = pyro.sample("v", dist.Normal(0, 1).expand([g, k]).to_event(2))

    alpha = torch.ones(h) * alpha_hparam
    alpha[0] = alpha0_hparam

    p = pyro.sample("p", dist.Dirichlet(alpha).expand([g, k]).to_event(2))

    p_logit = torch.logit(p)

    c = pyro.sample("c", dist.Normal(0, 1).expand([g, k]).to_event(2))

    theta = torch.einsum("gkh,ikh,gk->igk", p_logit, w, v)
    theta += c.squeeze()[None, ...]

    theta = pyro.deterministic("theta", theta.permute(2, 0, 1))

    y = pyro.sample(
        "y", dist.NegativeBinomial(expected_exp, logits=theta).to_event(3), obs=data
    )

    y_loc = y_h.sum(dim=0)

    y = pyro.sample("y", dist.Normal(y_loc, 1).to_event(2), obs=y)
