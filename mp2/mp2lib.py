import numpy as np
from scipy.special import erfc
from scipy.stats import norm


def inv_cdf_exponential(u, lam): # inverse CDF, exponential
    u = np.asarray(u, dtype=float)
    return -np.log1p(-u) / lam          # log1p(-u) = ln(1-u), accurate for small u


def inv_cdf_rayleigh(u, sigma): # inverse CDF, Rayleigh
    u = np.asarray(u, dtype=float)
    return sigma * np.sqrt(-2.0 * np.log1p(-u))


def sample_by_inversion(inv_cdf, n, rng, **kw): # uniform in, samples out
    return inv_cdf(rng.random(n), **kw)


def ks_statistic(samples, cdf): # sup |emperical CDF - analtical CDF|
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    F = cdf(x)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))


def q_function(x): # the Gaussian tail probability
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))



# CDF of Y = X^2
def cdf_y(y):
    y = np.asarray(y)
    return np.where(
        y > 0,
        2 * norm.cdf(np.sqrt(np.maximum(y, 0))) - 1,
        0
    )

# PDF of Y = X^2
def pdf_y(y):
    y = np.asarray(y)
    result = np.zeros_like(y, dtype=float)

    positive = y > 0

    result[positive] = (
        np.exp(-y[positive] / 2)
        / np.sqrt(2 * np.pi * y[positive])
    )

    return result



# Calculate the first four Rayleigh raw moments
def rayleigh_moments():
    from math import gamma

    moments = []
    for k in range(1, 5):
        moment = (2 ** (k / 2)) * gamma(1 + k / 2)
        moments.append(moment)

    return moments


# Generate Rayleigh and Pareto samples
def generate_samples(N, seed=42):
    rng = np.random.default_rng(seed)

    rayleigh = rng.rayleigh(scale=1, size=N)
    pareto = rng.pareto(a=1.5, size=N) + 1

    return rayleigh, pareto


# Calculate running mean and sample variance
def running_stats(samples, sizes):
    means = []
    variances = []

    for N in sizes:
        data = samples[:N]
        means.append(np.mean(data))
        variances.append(np.var(data, ddof=1))

    return means, variances
