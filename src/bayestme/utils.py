import numpy as np
from scipy.sparse import issparse, csc_matrix, vstack
from scipy.stats import poisson
from scipy.linalg import solve_triangular, cho_solve
from sksparse.cholmod import cholesky


def get_kth_order_trend_filtering_matrix(edge_adjacency_matrix, k):
    """
    Calculate the k-th order trend filtering matrix given the oriented edge
    incidence matrix and the value of k.

    :param edge_adjacency_matrix:
    :param k:
    :return:
    """
    if k < 0:
        raise Exception('k must be at least 0th order.')
    result = edge_adjacency_matrix
    for i in range(k):
        result = edge_adjacency_matrix.T.dot(result) if i % 2 == 0 else edge_adjacency_matrix.dot(result)
    return result


def sample_horseshoe_plus(size=1):
    a = 1 / np.random.gamma(0.5, 1, size=size)
    b = 1 / np.random.gamma(0.5, a)
    c = 1 / np.random.gamma(0.5, b)
    d = 1 / np.random.gamma(0.5, c)
    return d, c, b, a


def sample_mvn_from_precision(Q, mu=None, mu_part=None, sparse=True, chol_factor=False, Q_shape=None):
    """
    Fast sampling from a multivariate normal with precision parameterization.
    Supports sparse arrays.

    :param Q: input array
    :param mu: If provided, assumes the model is N(mu, Q^-1)
    :param mu_part: If provided, assumes the model is N(Q^-1 mu_part, Q^-1)
    :param sparse: If true, assumes we are working with a sparse Q
    :param chol_factor: If true, assumes Q is a (lower triangular) Cholesky decomposition of the precision matrix
    :param Q_shape: input array shape
    :return:
    """
    assert np.any([Q_shape is not None, not chol_factor, not sparse])
    if sparse:
        # Cholesky factor LL' = PQP' of the prior precision Q
        # where P is the permuation that reorders Q, the ordering of resulting L follows P
        factor = cholesky(Q) if not chol_factor else Q

        # Solve L'h = z ==> L'^-1 z = h, this is a sample from the prior.
        z = np.random.normal(size=Q.shape[0] if not chol_factor else Q_shape[0])
        # reorder h by the permatation used in cholesky(Q)
        result = factor.solve_Lt(z, False)[np.argsort(factor.P())]
        if mu_part is not None:
            # no need to reorder here since solve_A use the original Q
            result += factor.solve_A(mu_part)
        return result

    # Q is the precision matrix. Q_inv would be the covariance.
    # We care about Q_inv, not Q. It turns out you can sample from a MVN
    # using the precision matrix by doing LL' = Cholesky(Precision)
    # then the covariance part of the draw is just inv(L')z where z is
    # a standard normal.
    Lt = np.linalg.cholesky(Q).T if not chol_factor else Q.T
    z = np.random.normal(size=Q.shape[0])
    result = solve_triangular(Lt, z, lower=False)
    if mu_part is not None:
        result += cho_solve((Lt, False), mu_part)
    elif mu is not None:
        result += mu
    return result


def ilogit(x):
    return 1 / (1 + np.exp(-x))


def stable_softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=axis, keepdims=True)
    softmax = numerator / denominator
    return softmax


def sample_horseshoe_plus(size=1):
    a = 1 / np.random.gamma(0.5, 1, size=size)
    b = 1 / np.random.gamma(0.5, a)
    c = 1 / np.random.gamma(0.5, b)
    d = 1 / np.random.gamma(0.5, c)
    return d, c, b, a


def construct_edge_adjacency(neighbors):
    """
    Build the oriented edge-adjacency matrix from a list of (v1, v2) edges.

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


def construct_trendfilter(edge_adjacency_matrix, t, sparse=False):
    """
    Builds the t'th-order trend filtering matrix from an edge adjacency matrix.
    t=0 is the fused lasso / total variation matrix
    t=1 is the linear trend filtering / graph laplacian matrix
    t=2 is the quadratic trend filtering matrix
    etc.

    :param edge_adjacency_matrix:
    :param t:
    :param sparse:
    :return:
    """
    transformed_edge_adjacency_matrix = edge_adjacency_matrix.copy().astype('float')
    for i in range(t):
        if i % 2 == 0:
            transformed_edge_adjacency_matrix = edge_adjacency_matrix.T.dot(transformed_edge_adjacency_matrix)
        else:
            transformed_edge_adjacency_matrix = edge_adjacency_matrix.dot(transformed_edge_adjacency_matrix)

    if sparse:
        # Add a coordinate sparsity penalty
        extra = csc_matrix(np.eye(edge_adjacency_matrix.shape[1]))
    else:
        # Add a single independent node to make the matrix full rank
        extra = csc_matrix((np.array([1.]), (np.array([0], dtype=int), np.array([0], dtype=int))),
                           shape=(1, transformed_edge_adjacency_matrix.shape[1]))
    transformed_edge_adjacency_matrix = vstack([transformed_edge_adjacency_matrix, extra])
    return transformed_edge_adjacency_matrix


def composite_trendfilter(edge_adjacency_matrix, k, anchor=0, sparse=False):
    """

    :param edge_adjacency_matrix:
    :param k:
    :param anchor:
    :param sparse:
    :return:
    """
    if sparse:
        dbayes = np.eye(edge_adjacency_matrix.shape[1])
    else:
        # Start with the simple mu_1 ~ N(0, sigma)
        dbayes = np.zeros((1, edge_adjacency_matrix.shape[1]))
        dbayes[0, anchor] = 1

    if issparse(edge_adjacency_matrix):
        dbayes = csc_matrix(dbayes)

    # Add in the k'th order diffs
    for k in range(k + 1):
        kth_order_trend_filtering_matrix = get_kth_order_trend_filtering_matrix(edge_adjacency_matrix, k=k)
        if issparse(dbayes):
            dbayes = vstack([dbayes, kth_order_trend_filtering_matrix])
        else:
            dbayes = np.concatenate([dbayes, kth_order_trend_filtering_matrix], axis=0)

    return dbayes


def multinomial_rvs(count, p):
    """
    Sample from the multinomial distribution with multiple p vectors.
    * count must be an (n-1)-dimensional numpy array.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.
    The return value has the same shape as p.
    Taken from: https://stackoverflow.com/questions/55818845/fast-vectorized-multinomial-in-python
    """
    out = np.zeros(p.shape, dtype=int)
    count = count.copy()
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1] - 1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
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

    :param pos: a 2 x N matrix of coordinates
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
