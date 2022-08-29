import numpy as np

from typing import Optional
from scipy.sparse import issparse, csc_matrix, vstack
from scipy.stats import poisson


def ilogit(x):
    return 1 / (1 + np.exp(-x))


def stable_softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=axis, keepdims=True)
    softmax = numerator / denominator
    return softmax


def sample_horseshoe_plus(size=1, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = np.random.default_rng()

    a = 1 / rng.gamma(0.5, 1, size=size)
    b = 1 / rng.gamma(0.5, a)
    c = 1 / rng.gamma(0.5, b)
    d = 1 / rng.gamma(0.5, c)
    return d, c, b, a


def is_first_order_discrete_difference_operator(a):
    """
    Test if a matrix is a discrete difference operator, meaning each row has one pair of
    (-1, 1) values representing adjacency in a graph structure.

    :type a: Matrix to test
    :return: True if matrix is a first order discrete difference operator, False otherwise.
    """
    if issparse(a):
        return (
                np.all(np.sum(a, axis=1).flatten() == 0) and
                np.all(np.max(a, axis=1).todense().flatten() == 1) and
                np.all(np.min(a, axis=1).todense().flatten() == -1)
        )
    else:
        return (
                np.all(np.sum(a, axis=1) == 0) and
                np.all(np.max(a, axis=1) == 1) and
                np.all(np.min(a, axis=1) == -1)
        )


def get_kth_order_discrete_difference_operator(first_order_discrete_difference_operator, k):
    """
    Calculate the k-th order trend filtering matrix given a first order discrete difference operator of shape
    M x N.

    :param first_order_discrete_difference_operator: Input first order discrete difference operator
    :param k: Order of the output discrete difference operator
    :return: If k is even, this returns an M x N size matrix, if k is odd, this returns an N x N size matrix
    """
    if not is_first_order_discrete_difference_operator(first_order_discrete_difference_operator):
        raise ValueError('Expected edge_adjacency_matrix to be a '
                         'first order discrete difference operator, instead got: {}'.format(
            first_order_discrete_difference_operator))

    if k < 0:
        raise ValueError('k must be at least 0th order.')

    result = first_order_discrete_difference_operator
    for i in range(k):
        result = first_order_discrete_difference_operator.T.dot(
            result) if i % 2 == 0 else first_order_discrete_difference_operator.dot(result)
    return result


def construct_edge_adjacency(neighbors):
    """
    Build the oriented edge-adjacency matrix in "discrete difference operator"
    form an interable of (v1, v2) tuples representing edges.

    :param neighbors: E x 2 array, where E is the number of edges
    :return: An E x V sparse matrix, where E is the number of edges and V the number of vertices
    """
    data, rows, cols = [], [], []
    nrows = 0
    for i, j in neighbors:
        data.extend([1, -1])
        rows.extend([nrows, nrows])
        cols.extend([i, j])
        nrows += 1
    edge_adjacency_matrix = csc_matrix((data, (rows, cols)))
    return edge_adjacency_matrix


def construct_trendfilter(adjacency_matrix, k, sparse=False):
    """
    Builds the k'th-order trend filtering matrix from an adjacency matrix.
    k=0 is the fused lasso / total variation matrix
    k=1 is the linear trend filtering / graph laplacian matrix
    k=2 is the quadratic trend filtering matrix etc.

    :param adjacency_matrix: An adjacency matrix in first order discrete difference operator form.
    :param k: Order of trend filtering
    :param sparse: If true return a sparse matrix, otherwise return a dense np.ndarray
    :return: Graph trend filtering matrix
    """
    if not is_first_order_discrete_difference_operator(adjacency_matrix):
        raise ValueError('Expected edge_adjacency_matrix to be a '
                         'first order discrete difference operator, instead got: {}'.format(adjacency_matrix))

    transformed_edge_adjacency_matrix = adjacency_matrix.copy().astype('float')
    for i in range(k):
        if i % 2 == 0:
            transformed_edge_adjacency_matrix = adjacency_matrix.T.dot(transformed_edge_adjacency_matrix)
        else:
            transformed_edge_adjacency_matrix = adjacency_matrix.dot(transformed_edge_adjacency_matrix)

    if sparse:
        # Add a coordinate sparsity penalty
        extra = csc_matrix(np.eye(adjacency_matrix.shape[1]))
    else:
        # Add a single independent node to make the matrix full rank
        extra = csc_matrix(
            (np.array([1.]), (np.array([0], dtype=int), np.array([0], dtype=int))),
            shape=(1, transformed_edge_adjacency_matrix.shape[1]))
    transformed_edge_adjacency_matrix = vstack([transformed_edge_adjacency_matrix, extra])
    return transformed_edge_adjacency_matrix


def construct_composite_trendfilter(adjacency_matrix, k, anchor=0, sparse=False):
    """
    Build the k^1 through k^n trendfilter matrices stacked on top of each other
    in order to penalize multiple k values

    :param adjacency_matrix: An adjacency matrix in first order discrete difference operator form.
    :param k: Maximum order of trend filtering.
    :param anchor: Node index to set to 1 in the first row of the resulting matrix
    :param sparse: If true return a sparse matrix.
    :return: Composite trendfilter matrix.
    """
    if not is_first_order_discrete_difference_operator(adjacency_matrix):
        raise ValueError('Expected edge_adjacency_matrix to be a '
                         'first order discrete difference operator, instead got: {}'.format(adjacency_matrix))

    if sparse:
        composite_trendfilter_matrix = np.eye(adjacency_matrix.shape[1])
    else:
        # Start with the simple mu_1 ~ N(0, sigma)
        composite_trendfilter_matrix = np.zeros((1, adjacency_matrix.shape[1]))
        composite_trendfilter_matrix[0, anchor] = 1

    if issparse(adjacency_matrix):
        composite_trendfilter_matrix = csc_matrix(composite_trendfilter_matrix)

    # Add in the k'th order diffs
    for k in range(k + 1):
        kth_order_trend_filtering_matrix = get_kth_order_discrete_difference_operator(adjacency_matrix, k=k)
        if issparse(composite_trendfilter_matrix):
            composite_trendfilter_matrix = vstack(
                [composite_trendfilter_matrix,
                 kth_order_trend_filtering_matrix])
        else:
            composite_trendfilter_matrix = np.concatenate(
                [composite_trendfilter_matrix,
                 kth_order_trend_filtering_matrix], axis=0)

    return composite_trendfilter_matrix


def multinomial_rvs(count, p, rng: Optional[np.random.Generator] = None):
    """
    Sample from the multinomial distribution with multiple p vectors.
    * count must be an (n-1)-dimensional numpy array.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.
    The return value has the same shape as p.
    Taken from: https://stackoverflow.com/questions/55818845/fast-vectorized-multinomial-in-python
    """
    if rng is None:
        rng = np.random.default_rng()

    out = np.zeros(p.shape, dtype=int)
    count = count.copy()
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1] - 1, 0, -1):
        binsample = rng.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out


def logp(beta, components, attr, obs):
    attribute = beta * attr
    lams = np.einsum('nk,kg->nkg', attribute, components)
    lams = np.clip(lams.sum(axis=1), 1e-6, None)
    p = poisson.logpmf(obs, lams).sum()
    return p


def log_likelihood(attrributes, n_cells, Obs):
    # attrributes = beta[:, None] * components          K by G
    # n_cell: type-wise cell num. in each spot d_ik     N by K
    lams = n_cells[:, :, None] * attrributes[None]
    lams = np.clip(lams.sum(axis=1), 1e-6, None)
    log_likelihood = poisson.logpmf(Obs, lams).sum()
    return log_likelihood


def get_posmap(pos):
    """
    Return a matrix where the (i, j) entry stores the idx of the reads
    at spatial coordinates (i, j) on the tissue sample.

    :param pos: A 2 x N matrix of coordinates.
    :return: An X x Y matrix of coordinates, where X and Y are
    the max coordinate values on their respective axes
    """
    pos_map = np.empty((pos[0].max() + 1, pos[1].max() + 1))
    pos_map[:, :] = np.nan
    for i in range(pos.shape[1]):
        x, y = pos[:, i]
        pos_map[x, y] = i
    return pos_map


def get_edges(pos, layout=1) -> np.ndarray:
    """
    Given a set of positions and plate layout, return adjacency edges.

    :param pos: An N x 2 array of coordinates
    :param layout: Plate layout enum
    :return: An <N edges> x 2 array
    """
    pos = pos.T
    pos_map = get_posmap(pos)
    edges = []
    if layout == 1:
        # If the current spot is '@' put edges between '@' and 'o's
        #  * *
        # * @ o
        #  o o
        for i in range(pos_map.shape[0]):
            for j in range(pos_map.shape[1]):
                if ~np.isnan(pos_map[i, j]):
                    if i + 1 < pos_map.shape[0]:
                        if j > 0 and ~np.isnan(pos_map[i + 1, j - 1]):
                            edges.append(np.array([pos_map[i, j], pos_map[i + 1, j - 1]]))
                        if j + 1 < pos_map.shape[1] and ~np.isnan(pos_map[i + 1, j + 1]):
                            edges.append(np.array([pos_map[i, j], pos_map[i + 1, j + 1]]))
                    if j + 2 < pos_map.shape[1] and ~np.isnan(pos_map[i, j + 2]):
                        edges.append(np.array([pos_map[i, j], pos_map[i, j + 2]]))
    elif layout == 2:
        # If the current spot is '@' put edges between '@' and 'o's
        # * * *
        # * @ o
        # * o *
        for i in range(pos_map.shape[0]):
            for j in range(pos_map.shape[1]):
                if ~np.isnan(pos_map[i, j]):
                    if i + 1 < pos_map.shape[0] and ~np.isnan(pos_map[i + 1, j]):
                        edges.append(np.array([pos_map[i, j], pos_map[i + 1, j]]))
                    if j + 1 < pos_map.shape[1] and ~np.isnan(pos_map[i, j + 1]):
                        edges.append(np.array([pos_map[i, j], pos_map[i, j + 1]]))
    else:
        raise RuntimeError('Unknown layout')

    edges = np.array(edges)
    edges = edges.astype(int)
    return edges


def filter_reads_to_top_n_genes(reads, n_gene):
    n_gene = min(n_gene, reads.shape[1])
    top = np.argsort(np.std(np.log(1 + reads), axis=0))[::-1]
    return reads[:, top[:n_gene]]


def get_stddev_ordering(reads: np.ndarray):
    return np.argsort(np.std(np.log(1 + reads), axis=0))[::-1]


def get_top_gene_names_by_stddev(
        reads: np.ndarray,
        gene_names: np.array,
        n_genes=int):
    ordering = get_stddev_ordering(reads)
    return gene_names[ordering][:n_genes]


def order_reads_by_stddev(reads: np.ndarray):
    ordering = get_stddev_ordering(reads)
    return reads[:, ordering]
