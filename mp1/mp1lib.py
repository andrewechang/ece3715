import numpy as np

def bsc(bits, p, rng):
    """
    Flip each bit independently with probability p.
    """
    bits = np.asarray(bits, dtype=int)
    flips = rng.random(bits.shape) < p
    return bits ^ flips.astype(int)