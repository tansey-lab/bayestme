from bayestme.data import SpatialExpressionDataset
from numpy.random import Generator
from typing import Optional
from bayestme.common import InferenceType


def sample_from_posterior(
    data: SpatialExpressionDataset,
    n_components: int = None,
    spatial_smoothing_parameter=None,
    n_samples=100,
    expression_truth=None,
    InferenceType=InferenceType.MCMC,
    rng: Optional[Generator] = None,
):
    pass
