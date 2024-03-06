import pyro
import pyro.distributions as dist
import torch


def model(expected_exp, y=None, k=None, h=None, alpha0_hparam=10, alpha_hparam=1):
    """
    Model for spatial expression
    :param expected_exp: Expected expression N spot x G genes

    """
    i = expected_exp.shape[0]
    g = expected_exp.shape[1]

    w = pyro.sample("w", dist.Normal(0, 1).expand([i, k, h]).to_event(3))

    v = pyro.sample("v", dist.Normal(0, 1).expand([g, k]).to_event(2))

    alpha = torch.ones(h) * alpha_hparam
    alpha[0] = alpha0_hparam

    p = pyro.sample("p", dist.Dirichlet(alpha).expand([g, k]).to_event(2))

    c = pyro.sample("c", dist.Normal(0, 1).expand([g, k]).to_event(2))

    theta = torch.einsum("gkh,ikh,gk->igk", p.squeeze(), w.squeeze(), v.squeeze())
    theta += c.squeeze()[None, ...]

    theta = pyro.deterministic("theta", theta.permute(2, 0, 1))

    pyro.sample(
        "y", dist.NegativeBinomial(expected_exp, logits=theta).to_event(3), obs=y
    )
