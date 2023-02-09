import numpy as np
import numpy.testing
import numpy.linalg

from bayestme import utils
from scipy.sparse import csc_matrix


def test_is_first_order_discrete_difference_operator():
    positive_input = np.array(
        [[1, -1, 0, 0, 0], [0, 1, -1, 0, 0], [0, 0, 1, -1, 0], [0, 0, 0, 1, -1]]
    )

    assert utils.is_first_order_discrete_difference_operator(positive_input)

    negative_input = np.array(
        [[1, -1, 0, 0, 1], [0, 1, -1, 0, 0], [0, 0, 1, -1, 0], [0, 0, 0, 1, -1]]
    )

    assert not utils.is_first_order_discrete_difference_operator(negative_input)

    sparse_positive_input = csc_matrix(positive_input)

    assert utils.is_first_order_discrete_difference_operator(sparse_positive_input)


def test_construct_edge_adjacency():
    edge_adjacency_matrix = utils.construct_edge_adjacency(
        np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    )
    numpy.testing.assert_equal(
        edge_adjacency_matrix.toarray(),
        np.array(
            [[1, -1, 0, 0, 0], [0, 1, -1, 0, 0], [0, 0, 1, -1, 0], [0, 0, 0, 1, -1]]
        ),
    )


def test_construct_composite_trendfilter():
    edge_adjacency_matrix = np.array(
        [[1, -1, 0, 0, 0], [0, 1, -1, 0, 0], [0, 0, 1, -1, 0], [0, 0, 0, 1, -1]]
    )

    result = utils.construct_composite_trendfilter(
        edge_adjacency_matrix, k=2, sparse=False
    )

    expected = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, -1, 0, 0, 0],
            [0, 1, -1, 0, 0],
            [0, 0, 1, -1, 0],
            [0, 0, 0, 1, -1],
            [1, -1, 0, 0, 0],
            [-1, 2, -1, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 0, -1, 2, -1],
            [0, 0, 0, -1, 1],
            [2, -3, 1, 0, 0],
            [-1, 3, -3, 1, 0],
            [0, -1, 3, -3, 1],
            [0, 0, -1, 3, -2],
        ]
    )

    numpy.testing.assert_equal(result, expected)


def test_construct_trendfilter():
    edge_adjacency_matrix = np.array(
        [[1, -1, 0, 0, 0], [0, 1, -1, 0, 0], [0, 0, 1, -1, 0], [0, 0, 0, 1, -1]]
    )

    result = utils.construct_trendfilter(edge_adjacency_matrix, k=2, sparse=False)

    expected = np.array(
        [
            [2, -3, 1, 0, 0],
            [-1, 3, -3, 1, 0],
            [0, -1, 3, -3, 1],
            [0, 0, -1, 3, -2],
            [1, 0, 0, 0, 0],
        ]
    )

    numpy.testing.assert_equal(result.todense(), expected)

    # assert matrix is full rank
    assert numpy.linalg.matrix_rank(result.todense(), 5)


def test_get_kth_order_trend_filtering_matrix():
    adjacency_matrix = np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]])
    result = utils.get_kth_order_discrete_difference_operator(adjacency_matrix, k=2)

    expected = np.array([[2, -3, 1, 0], [-1, 3, -3, 1], [0, -1, 3, -2]])

    numpy.testing.assert_equal(result, expected)


def test_get_stddev_ordering():
    raw_counts = np.array(
        [[1, 199, 1, 1], [1, 10000, 500, 2], [1, 1, 3, 3]], dtype=np.int64
    )
    result = utils.get_stddev_ordering(raw_counts)
    np.testing.assert_equal(np.array([1, 2, 3, 0]), result)


def test_get_top_gene_names_by_stddev():
    raw_counts = np.array(
        [[1, 199, 1, 1], [1, 10000, 500, 2], [1, 1, 3, 3]], dtype=np.int64
    )
    result = utils.get_top_gene_names_by_stddev(
        reads=raw_counts,
        gene_names=np.array(["bad1", "best", "second_best", "x"]),
        n_genes=2,
    )

    np.testing.assert_equal(np.array(["best", "second_best"]), result)
