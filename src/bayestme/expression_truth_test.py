import numpy as np

from bayestme import expression_truth


def test_combine_multiple_expression_truth():
    data = [
        np.array([[1, 2, 3, 0], [2, 3, 4, 1], [2, 2, 2, 0]]),
        np.array([[1, 2, 4, 0], [2, 3, 6, 1], [2, 1, 2, 0]]),
    ]

    result = expression_truth.combine_multiple_expression_truth(
        expression_truth_arrays=data, num_warmup=10, num_samples=10
    )

    assert result.shape == (3, 4)
    assert np.all(result > 0)
