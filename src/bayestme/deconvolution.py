from bayestme.data import SpatialExpressionDataset
from numpy.random import Generator
from typing import Optional
from bayestme.common import InferenceType
from bayestme import data
import bayestme.mcmc.deconvolution
import bayestme.svi.deconvolution


def sample_from_posterior(
    data: SpatialExpressionDataset,
    n_components: int = None,
    spatial_smoothing_parameter=None,
    n_samples=100,
    expression_truth=None,
    inference_type=InferenceType.MCMC,
    rng: Optional[Generator] = None,
) -> data.DeconvolutionResult:
    if inference_type == InferenceType.MCMC:
        return bayestme.mcmc.deconvolution.deconvolve(
            reads=data.reads,
            edges=data.edges,
            n_samples=n_samples,
            n_burnin=1000,
            n_thin=5,
            n_gene=data.n_gene,
            n_components=n_components,
            lam2=spatial_smoothing_parameter,
            expression_truth=expression_truth,
            rng=rng,
        )
    elif inference_type == InferenceType.SVI:
        return bayestme.svi.deconvolution.deconvolve(
            stdata=data,
            n_components=n_components,
            rho=spatial_smoothing_parameter,
            n_samples=n_samples,
            rng=rng,
        )
    else:
        raise ValueError()
