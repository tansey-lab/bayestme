import numpy as np
from bayestme import fast_multivariate_normal


def test_sample_mvn_from_precision():
    Q = np.array([[1, 0.4], [0.4, 1]])

    x = fast_multivariate_normal.sample_multivariate_normal_from_precision(Q, sparse=False, chol_factor=False)

    assert x.shape == (2,)


def test_sample_mvn_from_covariance():
    Q = np.array([[1, 0.4], [0.4, 1]])

    x = fast_multivariate_normal.sample_multivariate_normal_from_covariance(Q, sparse=False, chol_factor=False)

    assert x.shape == (2,)
