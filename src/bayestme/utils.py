import numpy as np
from collections import defaultdict
from scipy.sparse import issparse, coo_matrix, csc_matrix, vstack
from scipy.stats import poisson
from scipy.linalg import solve_triangular, cho_solve


def hypercube_edges(dims, use_map=False):
    '''Create edge lists for an arbitrary hypercube. TODO: this is probably not the fastest way.'''
    edges = []
    nodes = np.arange(np.product(dims)).reshape(dims)
    for i, d in enumerate(dims):
        for j in range(d - 1):
            for n1, n2 in zip(np.take(nodes, [j], axis=i).flatten(), np.take(nodes, [j + 1], axis=i).flatten()):
                edges.append((n1, n2))
    if use_map:
        return edge_map_from_edge_list(edges)
    return edges


def edge_map_from_edge_list(edges):
    result = defaultdict(list)
    for s, t in edges:
        result[s].append(t)
        result[t].append(s)
    return result


def matrix_from_edges(edges):
    '''Returns a sparse penalty matrix (D) from a list of edge pairs. Each edge
    can have an optional weight associated with it.'''
    max_col = 0
    cols = []
    rows = []
    vals = []
    if type(edges) is defaultdict:
        edge_list = []
        for i, neighbors in edges.items():
            for j in neighbors:
                if i <= j:
                    edge_list.append((i, j))
        edges = edge_list
    for i, edge in enumerate(edges):
        s, t = edge[0], edge[1]
        weight = 1 if len(edge) == 2 else edge[2]
        cols.append(min(s, t))
        cols.append(max(s, t))
        rows.append(i)
        rows.append(i)
        vals.append(weight)
        vals.append(-weight)
        if cols[-1] > max_col:
            max_col = cols[-1]
    return coo_matrix((vals, (rows, cols)), shape=(rows[-1] + 1, max_col + 1)).tocsc()


def get_delta(D, k):
    '''Calculate the k-th order trend filtering matrix given the oriented edge
    incidence matrix and the value of k.'''
    if k < 0:
        raise Exception('k must be at least 0th order.')
    result = D
    for i in range(k):
        result = D.T.dot(result) if i % 2 == 0 else D.dot(result)
    return result


def grid_penalty_matrix(dims, k):
    edges = hypercube_edges(dims)
    D = matrix_from_edges(edges)
    return get_delta(D, k)


def sample_horseshoe_plus(size=1):
    a = 1 / np.random.gamma(0.5, 1, size=size)
    b = 1 / np.random.gamma(0.5, a)
    c = 1 / np.random.gamma(0.5, b)
    d = 1 / np.random.gamma(0.5, c)
    return d, c, b, a


def sample_horseshoe(size=1):
    a = 1 / np.random.gamma(0.5, 1, size=size)
    return 1 / np.random.gamma(0.5, a), a


def sample_mvn_from_precision(Q, mu=None, mu_part=None, sparse=True, chol_factor=False, Q_shape=None):
    '''Fast sampling from a multivariate normal with precision parameterization.
    Supports sparse arrays. Params:
        - mu: If provided, assumes the model is N(mu, Q^-1)
        - mu_part: If provided, assumes the model is N(Q^-1 mu_part, Q^-1)
        - sparse: If true, assumes we are working with a sparse Q
        - chol_factor: If true, assumes Q is a (lower triangular) Cholesky
                        decomposition of the precision matrix
    '''
    from sksparse.cholmod import cholesky
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


def logit(x):
    return np.log(x / (1 - x))


def sigmoid(x, inverse=False):
    if inverse:
        return (1 - x) / x
    else:
        return x / (1 - x)


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
    '''Builds the oriented edge-adjacency matrix from a list of (v1, v2) edges.'''
    from scipy.sparse import csc_matrix
    data, rows, cols = [], [], []
    nrows = 0
    for i, j in neighbors:
        # if i < j:
        data.extend([1, -1])
        rows.extend([nrows, nrows])
        cols.extend([i, j])
        nrows += 1
    D = csc_matrix((data, (rows, cols)))
    return D


def construct_trendfilter(D, t, eps=1e-4, sparse=False):
    '''Builds the t'th-order trend filtering matrix from an edge adjacency matrix.
    t=0 is the fused lasso / total variation matrix
    t=1 is the linear trend filtering / graph laplacian matrix
    t=2 is the quadratic trend filtering matrix
    etc.
    '''
    from scipy.sparse import vstack, csc_matrix
    Delta = D.copy().astype('float')
    for i in range(t):
        if i % 2 == 0:
            Delta = D.T.dot(Delta)
        else:
            Delta = D.dot(Delta)

    # Add a small amount of independent noise to make the matrix full rank
    # Delta[np.arange(min(Delta.shape)), np.arange(min(Delta.shape))] += eps

    if sparse:
        # Add a coordinate sparsity penalty
        extra = csc_matrix(np.eye(D.shape[1]))
    else:
        # Add a single independent node to make the matrix full rank
        extra = csc_matrix((np.array([1.]), (np.array([0], dtype=int), np.array([0], dtype=int))),
                           shape=(1, Delta.shape[1]))
    Delta = vstack([Delta, extra])
    return Delta


def composite_trendfilter(D, K, anchor=0, sparse=False):
    if sparse:
        Dbayes = np.eye(D.shape[1])
    else:
        # Start with the simple mu_1 ~ N(0, sigma)
        Dbayes = np.zeros((1, D.shape[1]))
        Dbayes[0, anchor] = 1

    if issparse(D):
        Dbayes = csc_matrix(Dbayes)

    # Add in the k'th order diffs
    for k in range(K + 1):
        Dk = get_delta(D, k=k)
        if issparse(Dbayes):
            Dbayes = vstack([Dbayes, Dk])
        else:
            Dbayes = np.concatenate([Dbayes, Dk], axis=0)

    return Dbayes


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


def sigma(p):
    return 1 / (1 + np.exp(-p))


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
    ### get a matrix where the (i, j) entry stores the idx of the reads at spatial coordinates (i, j) on the tissue sample
    ### pos shape 2 by N
    pos_map = np.empty((pos[0].max() + 1, pos[1].max() + 1))
    pos_map[:, :] = np.nan
    for i in range(pos.shape[1]):
        x, y = pos[:, i]
        pos_map[x, y] = i
    return pos_map


def get_edges(pos, layout=1):
    ### get edge graph from spot position and layout
    ### layout  1 = Visium (hex)
    ###         2 = ST (square)
    ### pos shape 2 by N
    pos_map = get_posmap(pos)
    edges = []
    if layout == 1:
        # current spot as '@' put edges between '@' and 'o's 
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
        # current spot as '@' put edges between '@' and 'o's 
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
        raise Exception('Unknown layout')

    edges = np.array(edges)
    edges = edges.astype(int)
    return edges


def DIC(cell_post, beta_post, components_post, cell_attributes_post, Observations_tissue, idx):
    N = 0
    dic = 0
    for n in idx:
        loglh = logp(cell_post[-n], beta_post[-n], components_post[-n].T, cell_attributes_post[-n], Observations_tissue)
        if ~np.isnan(loglh):
            dic += -2 * loglh
            N += 1
    dic /= N
    dic *= 2
    dic -= -2 * logp(cell_post[-idx].mean(axis=0), beta_post[-idx].mean(axis=0), components_post[-idx].mean(axis=0).T,
                     cell_attributes_post[-idx].mean(axis=0), Observations_tissue)
    return dic


def get_WAIC(beta, components, attr, obs, idx):
    attribute = beta_trace[-idx, None] * cell_assignment_num_trace[-idx]
    lams = attribute[:, :, :, None] * gene_expression_trace[-idx, None]
    lams = np.clip(lams.sum(axis=2), 1e-6, None)
    # lppd
    pd = np.clip(poisson.pmf(Observation[None], lams), 1e-6, None)
    pd_mean = pd.mean(axis=0)
    lppd = np.log(pd_mean).sum()
    # p_waic
    loglikelihood = np.log(pd)
    mean_ll = loglikelihood.mean(axis=0)
    v_s = ((loglikelihood - mean_ll[None, :]) ** 2).sum(axis=0) / (loglikelihood.shape[0] - 1)
    p_waic = v_s.sum()
    # scale by -2 (Gelman et al. 2013)
    waic = -2 * (lppd - p_waic)
    return waic


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
