import numpy as np
import matplotlib.pyplot as plt

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

def sensor_covariances(c):
    Sigma_N = np.array([
        [1.0, c],
        [c, 2.0]
    ])

    Sigma_xx = 4.0 * np.ones((2, 2)) + Sigma_N
    Sigma_sx = np.array([4.0, 4.0])

    return Sigma_N, Sigma_xx, Sigma_sx

def sensor_lmmse(c):
    Sigma_N, Sigma_xx, Sigma_sx = sensor_covariances(c)

    if not is_valid_covariance(Sigma_N):
        raise ValueError("Invalid noise covariance")

    w = lmmse_weights(Sigma_xx, Sigma_sx)

    mmse = 4.0 - Sigma_sx @ w

    return w, float(mmse)

def sensor_monte_carlo(c, rng, n_trials=1_000_000):
    Sigma_N, _, _ = sensor_covariances(c)
    w, analytic_mmse = sensor_lmmse(c)

    total_error = 0.0
    batch_size = 100_000

    for start in range(0, n_trials, batch_size):
        n = min(batch_size, n_trials - start)

        S = rng.normal(0, 2, size=n)

        N = rng.multivariate_normal(
            [0, 0], Sigma_N, size=n
        )

        X = S[:, None] + N
        S_hat = X @ w

        total_error += np.sum((S - S_hat)**2)

    simulated_mse = total_error / n_trials

    return simulated_mse, analytic_mmse

def cholesky_samples(Sigma, n_samples, rng):
    Sigma = np.asarray(Sigma, dtype=float)

    L = np.linalg.cholesky(Sigma)
    z = rng.standard_normal((n_samples, Sigma.shape[0]))

    return z @ L.T

def eigen_samples(Sigma, n_samples, rng):
    Sigma = np.asarray(Sigma, dtype=float)

    eigenvalues, V = np.linalg.eigh(Sigma)

    if np.any(eigenvalues < -1e-10):
        raise ValueError("Covariance matrix must be positive semidefinite")

    eigenvalues = np.maximum(eigenvalues, 0)

    A = V @ np.diag(np.sqrt(eigenvalues))
    z = rng.standard_normal((n_samples, Sigma.shape[0]))

    return z @ A.T

def verify_covariance(Sigma, n_samples, rng):
    X_chol = cholesky_samples(Sigma, n_samples, rng)
    X_eig = eigen_samples(Sigma, n_samples, rng)

    _, cov_chol = sample_mean_cov(X_chol)
    _, cov_eig = sample_mean_cov(X_eig)

    return cov_chol, cov_eig

def standardize_data(X):
    X = np.asarray(X, dtype=float)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)

    if np.any(std == 0):
        raise ValueError("Cannot standardize a constant column")

    return (X - mean) / std

def pca_analysis(X):
    X = np.asarray(X, dtype=float)

    # Calculate mean and covariance matrix
    mean, covariance = sample_mean_cov(X)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort eigenvalues from largest to smallest
    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    # Fraction of total variance
    explained_variance = eigenvalues / np.sum(eigenvalues)

    # Project data onto first two components
    centered = X - mean
    projected = centered @ eigenvectors[:, :2]

    return covariance, eigenvalues, eigenvectors, explained_variance, projected

def plot_pca(projected, labels, title):
    plt.figure(figsize=(8, 6))

    for i in np.unique(labels):
        plt.scatter(
            projected[labels == i, 0],
            projected[labels == i, 1],
            label=f"Class {i}",
            alpha=0.7
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()