import numpy as np

def bsc(bits, p, rng):
    """
    Flip each bit independently with probability p.
    """
    bits = np.asarray(bits, dtype=int)
    flips = rng.random(bits.shape) < p
    return bits ^ flips.astype(int)


def packet_errors(n_pkt, n_bits, p, rng, chunk=5000):
    """
    Number of bit errors in each of n_pkt packets of n_bits bits.
    Every bit goes through bsc (all-zero packets, so a 1 is an error).
    Packets are processed in chunks to keep memory small.
    """
    out = np.empty(n_pkt, dtype=int)
    for start in range(0, n_pkt, chunk):
        m = min(chunk, n_pkt - start)
        out[start:start + m] = bsc(np.zeros((m, n_bits), dtype=int), p, rng).sum(axis=1)
    return out


def tv_distance(pmf1, pmf2):
    """
    Total variation distance: HALF the sum of |pmf1[k] - pmf2[k]|.
    Both arrays must be over the same support.
    """
    pmf1 = np.asarray(pmf1, dtype=float)
    pmf2 = np.asarray(pmf2, dtype=float)
    return 0.5 * np.sum(np.abs(pmf1 - pmf2))

def attempts_until_clean(n_trials, q, rng):
    """
    Draw the number of attempts until the first clean packet.
    The support is 1, 2, 3, ...
    q is the per-attempt clean probability.
    """
    if n_trials < 0 or not 0 < q <= 1:
        raise ValueError("n_trials must be nonnegative and q must be in (0, 1]")

    return rng.geometric(q, size=n_trials)

def run_step7_experiments(N_values, num_runs=200, seed=42):
    """
    Run repeated Binomial and Geometric experiments and calculate
    the running sample means at each value of N.
    """

    rng = np.random.default_rng(seed)

    # Distribution parameters
    binom_n = 1000
    binom_p = 0.005
    geom_p = 0.7

    N_max = N_values[-1]

    # Storage
    binom_means = np.zeros((num_runs, len(N_values)))
    geom_means = np.zeros((num_runs, len(N_values)))

    for run in range(num_runs):

        # Binomial samples
        binom_samples = rng.binomial(
            binom_n,
            binom_p,
            size=N_max
        )

        binom_cumsum = np.cumsum(binom_samples)

        binom_means[run, :] = (
            binom_cumsum[N_values - 1] / N_values
        )

        # Geometric samples
        geom_samples = rng.geometric(
            geom_p,
            size=N_max
        )

        geom_cumsum = np.cumsum(geom_samples)

        geom_means[run, :] = (
            geom_cumsum[N_values - 1] / N_values
        )

    return binom_means, geom_means


def calculate_step7_spread(means):
    """
    Calculate the 5th percentile, 95th percentile,
    and mean across all experiments.
    """

    low = np.percentile(means, 5, axis=0)
    high = np.percentile(means, 95, axis=0)
    center = np.mean(means, axis=0)

    return low, high, center


def calculate_envelope(N_values, low, high, match_idx=10):
    """
    Calculate a 1/sqrt(N) envelope matched to the
    observed spread at one checkpoint.
    """

    half_width = (
        high[match_idx] - low[match_idx]
    ) / 2

    C = half_width * np.sqrt(N_values[match_idx])

    envelope = C / np.sqrt(N_values)

    return envelope


def find_trials_within_one_percent(
    N_values,
    low,
    high,
    true_mean
):
    """
    Find the first checkpoint where the entire spread band
    is within 1% of the true mean.
    """

    tolerance = 0.01 * true_mean

    lower_target = true_mean - tolerance
    upper_target = true_mean + tolerance

    within_1_percent = (
        (low >= lower_target) &
        (high <= upper_target)
    )

    if np.any(within_1_percent):
        first_idx = np.where(within_1_percent)[0][0]
        needed_N = N_values[first_idx]

        return (
            needed_N,
            first_idx,
            tolerance,
            lower_target,
            upper_target
        )

    return (
        None,
        None,
        tolerance,
        lower_target,
        upper_target
    )