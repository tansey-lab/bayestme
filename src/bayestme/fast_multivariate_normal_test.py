import numpy as np
import numpy.random
import numpy.testing

from bayestme import fast_multivariate_normal


def test_sample_mvn_from_precision():
    Q = np.array([[1, 0.4], [0.4, 1]])

    x = fast_multivariate_normal.sample_multivariate_normal_from_precision(
        Q, sparse=False, chol_factor=False
    )

    assert x.shape == (2,)


def test_sample_stability():
    Q = np.array([[1, 0.4], [0.4, 1]])

    x_1 = fast_multivariate_normal.sample_multivariate_normal_from_precision(
        Q, sparse=False, chol_factor=False, rng=numpy.random.default_rng(1)
    )
    x_2 = fast_multivariate_normal.sample_multivariate_normal_from_precision(
        Q, sparse=False, chol_factor=False, rng=numpy.random.default_rng(1)
    )
    x_3 = fast_multivariate_normal.sample_multivariate_normal_from_precision(
        Q, sparse=False, chol_factor=False, rng=numpy.random.default_rng(1)
    )
    numpy.testing.assert_array_equal(x_1, x_2)

    numpy.testing.assert_array_equal(x_2, x_3)


def test_sequence_stability():
    Q = np.array([[1, 0.4], [0.4, 1]])
    rng = numpy.random.default_rng(1)

    trial_one = [
        fast_multivariate_normal.sample_multivariate_normal_from_precision(
            Q, sparse=False, chol_factor=False, rng=rng
        )
        for _ in range(10)
    ]
    rng = numpy.random.default_rng(1)
    trial_two = [
        fast_multivariate_normal.sample_multivariate_normal_from_precision(
            Q, sparse=False, chol_factor=False, rng=rng
        )
        for _ in range(10)
    ]

    for a, b in zip(trial_one, trial_two):
        numpy.testing.assert_array_equal(a, b)
