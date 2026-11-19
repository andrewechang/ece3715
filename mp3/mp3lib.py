import numpy as np

def sample_mean_cov(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array with shape (N, d)")
    n_samples, n_variables = X.shape
    if n_samples < 2:
        raise ValueError("at least two samples are required")

    mean_vector = np.mean(X, axis=0)
    centered = X - mean_vector
    cov_matrix = (centered.T @ centered) / (n_samples - 1)

    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
    return mean_vector, cov_matrix


def is_valid_covariance(S, tol=1e-10):
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        return False
    if not np.all(np.isfinite(S)):
        return False
    if not np.allclose(S, S.T, atol=tol, rtol=0.0):
        return False

    eigenvalues = np.linalg.eigh(0.5 * (S + S.T))[0]
    return bool(np.min(eigenvalues) >= -tol)


def gaussian_pdf(x, mu, Sigma):
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    if mu.ndim != 1:
        raise ValueError("mu must be a one-dimensional vector")
    dimension = mu.size
    if x.ndim == 0 or x.shape[-1] != dimension:
        raise ValueError("the final axis of x must have the same length as mu")
    if Sigma.shape != (dimension, dimension):
        raise ValueError("Sigma must have shape (d, d)")
    if not np.allclose(Sigma, Sigma.T, atol=1e-12, rtol=0.0):
        raise ValueError("Sigma must be symmetric")

    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        raise ValueError("Sigma must be positive definite")

    delta = x - mu
    quadratic = np.sum(delta * np.linalg.solve(Sigma, delta[..., np.newaxis])[..., 0], axis=-1)
    normalization_log = -0.5 * (dimension * np.log(2.0 * np.pi) + logdet)
    return np.exp(normalization_log - 0.5 * quadratic)


def whitening_matrix(Sigma):
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be a square matrix")
    if not np.allclose(Sigma, Sigma.T, atol=1e-12, rtol=0.0):
        raise ValueError("Sigma must be symmetric")

    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    if np.min(eigenvalues) <= 0:
        raise ValueError("Sigma must be positive definite to whiten")

    inverse_sqrt = 1.0 / np.sqrt(eigenvalues)
    return (eigenvectors * inverse_sqrt) @ eigenvectors.T


def lmmse_weights(Sigma_xx, Sigma_sx):
    Sigma_xx = np.asarray(Sigma_xx, dtype=float)
    Sigma_sx = np.asarray(Sigma_sx, dtype=float)

    if Sigma_xx.ndim != 2 or Sigma_xx.shape[0] != Sigma_xx.shape[1]:
        raise ValueError("Sigma_xx must be square")
    dimension = Sigma_xx.shape[0]
    if Sigma_sx.shape not in {(dimension,), (dimension, 1)}:
        raise ValueError("Sigma_sx must have shape (d,) or (d, 1)")

    weights = np.linalg.solve(Sigma_xx, Sigma_sx.reshape(dimension, -1))
    return weights[:, 0]