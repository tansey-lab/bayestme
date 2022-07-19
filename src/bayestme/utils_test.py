import numpy as np
import numpy.testing

from bayestme import utils


def test_construct_edge_adjacency():
    edge_adjacency_matrix = utils.construct_edge_adjacency(np.array([[0, 1], [1, 2], [2, 3], [3, 4]]))
    numpy.testing.assert_equal(edge_adjacency_matrix.toarray(),
                               np.array([[1, -1, 0, 0, 0], [0, 1, -1, 0, 0], [0, 0, 1, -1, 0], [0, 0, 0, 1, -1]]))


def test_get_stddev_ordering():
    raw_counts = np.array(
        [[1, 199, 1, 1],
         [1, 10000, 500, 2],
         [1, 1, 3, 3]], dtype=np.int64
    )
    result = utils.get_stddev_ordering(
        raw_counts
    )
    np.testing.assert_equal(
        np.array([1, 2, 3, 0]),
        result
    )


def test_get_top_gene_names_by_stddev():
    raw_counts = np.array(
        [[1, 199, 1, 1],
         [1, 10000, 500, 2],
         [1, 1, 3, 3]], dtype=np.int64
    )
    result = utils.get_top_gene_names_by_stddev(
        reads=raw_counts,
        gene_names=np.array(['bad1', 'best', 'second_best', 'x']),
        n_genes=2
    )

    np.testing.assert_equal(
        np.array(['best', 'second_best']),
        result
    )
