"""ECE 3715 Mini-Project 2 helper functions."""
import numpy as np
from scipy.special import erfc


def inv_cdf_exponential(u, lam):
    """Inverse CDF of Exp(lam): F(x)=1-exp(-lam x)  =>  F^-1(u) = -ln(1-u)/lam.

    rng.random() returns u in [0,1), so 1-u is in (0,1] and the log is always
    finite. u=0 maps to x=0, which is in the support."""
    u = np.asarray(u, dtype=float)
    return -np.log1p(-u) / lam          # log1p(-u) = ln(1-u), accurate for small u


def inv_cdf_rayleigh(u, sigma):
    """Inverse CDF of Rayleigh(sigma): F(x)=1-exp(-x^2/(2 sigma^2))
    =>  F^-1(u) = sigma*sqrt(-2 ln(1-u)). Same u=0 safety as above."""
    u = np.asarray(u, dtype=float)
    return sigma * np.sqrt(-2.0 * np.log1p(-u))


def sample_by_inversion(inv_cdf, n, rng, **kw):
    """Draw n uniforms from rng.random() and push them through inv_cdf."""
    return inv_cdf(rng.random(n), **kw)


def ks_statistic(samples, cdf):
    """sup |empirical CDF - analytic CDF|, evaluated exactly at the jumps."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    F = cdf(x)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))


def q_function(x):
    """Gaussian tail probability Q(x) = P[N(0,1) > x] = 0.5*erfc(x/sqrt(2))."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))