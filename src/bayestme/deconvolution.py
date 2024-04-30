from typing import Optional

from numpy.random import Generator

import bayestme.svi.deconvolution
from bayestme import data
from bayestme.data import SpatialExpressionDataset


def sample_from_posterior(
    data: SpatialExpressionDataset,
    n_components: int = None,
    spatial_smoothing_parameter=None,
    n_samples=100,
    n_svi_steps=10_000,
    expression_truth=None,
    use_spatial_guide=True,
    rng: Optional[Generator] = None,
) -> data.DeconvolutionResult:
    return bayestme.svi.deconvolution.deconvolve(
        stdata=data,
        n_components=n_components,
        rho=spatial_smoothing_parameter,
        n_samples=n_samples,
        n_svi_steps=n_svi_steps,
        use_spatial_guide=use_spatial_guide,
        expression_truth=expression_truth,
        rng=rng,
    )
