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