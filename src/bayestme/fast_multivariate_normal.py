from typing import Optional
from warnings import warn

import numpy as np
import scipy as sp
from numpy.linalg import LinAlgError
from scipy.linalg import solve_triangular
from scipy.sparse import csc_matrix
from sksparse.cholmod import cholesky, CholmodNotPositiveDefiniteError


def sample_multivariate_normal_from_precision(
    Q,
    mu=None,
    mu_part=None,
    sparse=True,
    chol_factor=False,
    Q_shape=None,
    force_psd=False,
    force_psd_eps=1e-6,
    force_psd_attempts=4,
    rng: Optional[np.random.Generator] = None,
):
    """
    Fast sampling from a multivariate normal with precision parameterization.
    Supports sparse arrays.

    :param Q: Precision matrix
    :param mu: If provided, assumes the model is N(mu, Q^-1)
    :param mu_part: If provided, assumes the model is N(Q^-1 mu_part, Q^-1)
    :param sparse: If true, assumes we are working with a sparse Q
    :param chol_factor: If true, assumes Q is a (lower triangular) Cholesky
                        decomposition of the precision matrix
    :param Q_shape:
    :param force_psd: If true, attempts to force the precision matrix to
                      be positive definite adding a diagonal term.
    :param force_psd_eps: If force_psd is true, force_psd_eps is the frist value added to the diagonal
                          to force the precision matrix to be positive definite.
    :param force_psd_attempts: If force_psd is true, this is the number of attempts to force
                               the precision matrix to be positive definite. Each attempt a diagonal term that
                               is 10 times larger than the previous one is added.
    :param rng: numpy.random.Generator to use
    :return: One sample from the multivariate normal distribution
    """
    if rng is None:
        rng = np.random.default_rng()

    if sparse and not Q_shape:
        raise ValueError("Need to provide one of q_shape if sparse.")

    if not np.any([Q_shape is not None, not chol_factor, not sparse]):
        raise ValueError("Need to provide one of q_shape, chol_factor, or sparse.")

    attempt = 0
    eps = force_psd_eps

    while True:
        try:
            if sparse:
                # Cholesky factor LL' = PQP' of the prior precision Q
                # where P is the permuation that reorders Q, the ordering of resulting L follows P
                factor = cholesky(Q) if not chol_factor else Q

                # Solve L'h = z ==> L'^-1 z = h, this is a sample from the prior.
                z = rng.normal(size=Q.shape[0] if not chol_factor else Q_shape[0])

                # Reorder h by the permutation used in cholesky(Q).
                result = factor.solve_Lt(z, False)[np.argsort(factor.P())]
                if mu_part is not None:
                    # no need to reorder here since solve_A use the original Q
                    result += factor.solve_A(mu_part)
            else:
                # Q is the precision matrix. Q_inv would be the covariance.
                # We care about Q_inv, not Q. It turns out you can sample from a MVN
                # using the precision matrix by doing LL' = Cholesky(Precision)
                # then the covariance part of the draw is just inv(L')z where z is
                # a standard normal.
                # Ordering should be good here since linalg.cholesky solves LL'=Q
                Lt = np.linalg.cholesky(Q).T if not chol_factor else Q.T
                z = rng.normal(size=Q.shape[0])
                result = solve_triangular(Lt, z, lower=False)
                if mu_part is not None:
                    result += sp.linalg.cho_solve((Lt, False), mu_part)
                elif mu is not None:
                    result += mu
        except (CholmodNotPositiveDefiniteError, LinAlgError) as e:
            if force_psd and attempt < force_psd_attempts:
                Q = Q.copy()
                Q[np.diag_indices_from(Q)] += eps
                warn(f"Cholesky factorization failed, adding shrinkage {eps}.")
                attempt += 1
                eps *= 10
            else:
                warn(
                    f"Cholesky factorization failed, try setting force_psd=True or increasing attempts"
                )
                if attempt > force_psd_attempts:
                    raise Exception(
                        "Max attempts reached. Could not force matrix to be positive definite."
                    ) from e
        else:
            return result


def sample_multivariate_normal_from_covariance(
    Q,
    mu=None,
    mu_part=None,
    sparse=True,
    chol_factor=False,
    force_psd=False,
    force_psd_eps=1e-6,
    force_psd_attempts=4,
    rng: Optional[np.random.Generator] = None,
):
    """
    Fast sampling from a multivariate normal with covariance parameterization.
    Supports sparse arrays.

    :param Q: Covariance matrix
    :param mu: If provided, assumes the model is N(mu, Q)
    :param mu_part: If provided, assumes the model is N(Q mu_part, Q)
    :param sparse: If true, assumes we are working with a sparse Q
    :param chol_factor: If true, assumes Q is a (lower triangular) Cholesky
                        decomposition of the covariance matrix
    :param force_psd: If true, attempts to force the covariance-matrix to
                      be positive definite adding a diagonal term
    :param force_psd_eps: If force_psd is true, force_psd_eps is the frist value added to the diagonal
                          to force the covariance matrix to be positive definite.l
    :param force_psd_attempts: If force_psd is true, this is the number of attempts to force
                               the covariance matrix to be positive definite. Each attempt a diagonal term that
                               is 10 times larger than the previous one is added.
    :param rng: numpy.random.Generator to use
    :return: One sample from the multivariate normal distribution
    """
    if rng is None:
        rng = np.random.default_rng()

    attempt = 0
    eps = force_psd_eps

    while True:
        try:
            if sparse:
                # Cholesky factor LL' = Q of the covariance matrix Q
                if chol_factor:
                    factor = Q
                    Q = factor.L().dot(factor.L().T)
                else:
                    factor = cholesky(Q)

                # Get the sample as mu + Lz for z ~ N(0, I)
                z = rng.normal(size=Q.shape[0])
                result = factor.L().dot(z)[np.argsort(factor.P())]
                if mu_part is not None:
                    result += Q.dot(mu_part)
                elif mu is not None:
                    result += mu
            else:
                # Cholesky factor LL' = Q of the covariance matrix Q
                if chol_factor:
                    Lt = Q
                    Q = Lt.dot(Lt.T)
                else:
                    Lt = np.linalg.cholesky(Q)

                # Get the sample as mu + Lz for z ~ N(0, I)
                z = rng.normal(size=Q.shape[0])
                result = Lt.dot(z)
                if mu_part is not None:
                    result += Q.dot(mu_part)
                elif mu is not None:
                    result += mu
        except (CholmodNotPositiveDefiniteError, LinAlgError) as e:
            if force_psd and attempt < force_psd_attempts:
                Q = Q.copy()
                Q[np.diag_indices_from(Q)] += eps
                warn(f"Cholesky factorization failed, adding shrinkage {eps}.")
                attempt += 1
                eps *= 10
            else:
                warn(
                    f"Cholesky factorization failed, try setting force_psd=True or increasing attempts"
                )
                if attempt > force_psd_attempts:
                    raise Exception(
                        "Max attempts reached. Could not force matrix to be positive definite."
                    ) from e
        else:
            return result


def sample_multivariate_normal(
    Q,
    mu=None,
    mu_part=None,
    sparse=True,
    precision=False,
    chol_factor=False,
    Q_shape=None,
    **kwargs,
):
    """
    Fast sampling from a multivariate normal with covariance or precision
    parameterization. Supports sparse arrays.

    :param Q: covariance or precision matrix
    :param mu: If provided, assumes the model is N(mu, Q)
    :param mu_part: If provided, assumes the model is N(Q mu_part, Q)
    :param sparse: If true, assumes we are working with a sparse Q
    :param precision: If true, assumes Q is a precision matrix (inverse covariance)
    :param chol_factor: If true, assumes Q is a (lower triangular) Cholesky
                        decomposition of the covariance matrix
                        (or of the precision matrix if precision=True).
    :param Q_shape:
    :param kwargs:
    :return: One sample from the multivariate normal
    """
    if not np.any((mu is None, mu_part is None)):
        raise ValueError("mu and mu_part are mutually exclusive optional parameters")

    # If Q is a scalar or vector, consider it Q*I
    if not chol_factor:
        if np.isscalar(Q) or len(Q.shape) == 1:
            dim = len(mu) if mu is not None else len(mu_part)
            Q = np.eye(dim) * Q
            if sparse:
                Q = csc_matrix(Q)

    # Sample from the appropriate precision or covariance version
    if precision:
        return sample_multivariate_normal_from_precision(
            Q,
            mu=mu,
            mu_part=mu_part,
            sparse=sparse,
            chol_factor=chol_factor,
            Q_shape=Q_shape,
            **kwargs,
        )
    return sample_multivariate_normal_from_covariance(
        Q, mu=mu, mu_part=mu_part, sparse=sparse, chol_factor=chol_factor, **kwargs
    )
